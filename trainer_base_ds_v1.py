# coding=utf-8
#
# Copyright 2022 Shandong University Fangkai Jiao
#
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import logging
import os
import sys
from typing import Dict, Union

import deepspeed
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)

from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, unwrap_model, set_seed, note_best_checkpoint

logger: logging.Logger


# transformers.logging.set_verbosity_error()

def get_zero_stage(cfg: DictConfig):
    if hasattr(cfg, "zero_optimization"):
        return int(getattr(cfg.zero_optimization, "stage", 0))
    return 0


def get_state_dict(model: Union[deepspeed.DeepSpeedEngine, deepspeed.PipelineEngine], cfg: DictConfig):
    zero_stage = get_zero_stage(cfg)
    if zero_stage == 3:
        state_dict = model._zero3_consolidated_fp16_state_dict()
    else:
        state_dict = unwrap_model(model).state_dict()
    return state_dict


def save_model(model: Union[deepspeed.DeepSpeedEngine, deepspeed.PipelineEngine],
               cfg: DictConfig, output_dir: str, tokenizer: PreTrainedTokenizer = None, state_dict: Dict = None):
    unwrapped_model = unwrap_model(model)
    if state_dict is None:
        state_dict = get_state_dict(model, cfg)

    if cfg.local_rank in [-1, 0]:
        # TODO: Is it necessary to convert the state dict of float16 into float32?
        # for k in state_dict:
        #     if state_dict[k].dtype == torch.float16:
        #         state_dict[k] = state_dict[k].float()

        unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)


def forward_step(model, inputs: Dict[str, torch.Tensor]):
    # loss = model(**inputs)["loss"]
    outputs = model(**inputs)
    if isinstance(outputs, tuple):
        loss = outputs[0]
    else:
        loss = outputs.pop("loss")
    model.backward(loss)
    model.step()

    return loss.item(), outputs


def train(cfg, train_dataset, features, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        _dir_splits = cfg.output_dir.split('/')
        _log_dir = '/'.join([_dir_splits[0], 'runs'] + _dir_splits[1:])
        tb_writer = SummaryWriter(log_dir=_log_dir)
        tb_helper = hydra.utils.instantiate(cfg.summary_helper,
                                            writer=tb_writer) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_writer = None
        tb_helper = None

    cfg.train_batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
    train_sampler = RandomSampler(train_dataset) if cfg.local_rank == -1 else DistributedSampler(train_dataset)
    train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=cfg.train_batch_size,
                                  collate_fn=train_collator, num_workers=cfg.num_workers, pin_memory=True,
                                  prefetch_factor=cfg.prefetch_factor)

    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (len(train_dataloader) // cfg.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // cfg.gradient_accumulation_steps * cfg.num_train_epochs

    num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    if "extended_vocab" in cfg and cfg.extended_vocab:
        logger.info(f"Extended extra vocab size: {cfg.extended_vocab}")
        model.resize_token_embeddings(model.config.vocab_size + cfg.extended_vocab)

    ds_config = cfg.ds_cfg
    ds_config.scheduler.params.total_num_steps = t_total
    ds_config.scheduler.params.warmup_num_steps = num_warmup_steps
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    # FIXME: Not supported in CPUAdam? Open an issue.
    # no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': cfg.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]

    model, optimizer, _, scheduler = deepspeed.initialize(model=model,
                                                          model_parameters=model.parameters(),
                                                          config=ds_config)
    logger.info(optimizer.optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                cfg.train_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if cfg.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
        if cfg.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(epoch_iterator):
            # If training is continued from a checkpoint, fast forward
            # to the state of that checkpoint.
            if global_step < continue_from_global_step:
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    global_step += 1
                continue

            model.train()
            batch = batch_to_device(batch, cfg.device)

            loss, outputs = forward_step(model, batch)
            loss /= cfg.gradient_accumulation_steps

            tr_loss += loss
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                global_step += 1

                # Log metrics
                if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / cfg.logging_steps, global_step)
                    logging_loss = tr_loss

                    if tb_helper:
                        tb_helper(step=global_step, last_batch=batch, last_outputs=outputs)

                # Save model checkpoint
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                    if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, cfg, output_dir, tokenizer)

                # Evaluation
                if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    state_dict = get_state_dict(model, cfg)

                    if cfg.local_rank in [-1, 0]:
                        results = evaluate(cfg, model, tokenizer, prefix=str(global_step), _split="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar(f"eval/{key}", value, global_step)

                        sub_path = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                        flag = note_best_checkpoint(cfg, results, sub_path)
                        if cfg.save_best and flag:
                            save_model(model, cfg, cfg.output_dir, tokenizer, state_dict)
                            del state_dict

            if 0 < cfg.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            train_iterator.close()
            break

    if cfg.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(cfg, model, tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    dataset, features = load_and_cache_examples(cfg, tokenizer, _split=_split)

    if not os.path.exists(os.path.join(cfg.output_dir, prefix)):
        os.makedirs(os.path.join(cfg.output_dir, prefix))

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator)
    single_model_gpu = unwrap_model(model)
    single_model_gpu.get_eval_log(reset=True)
    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)

    model.eval()
    pred_list = []
    prob_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True):
        batch = batch_to_device(batch, cfg.device)
        with torch.no_grad():
            outputs = model(**batch)
            probs = outputs["logits"].softmax(dim=-1).detach().float().cpu()
            prob, pred = probs.max(dim=-1)
            pred_list.extend(pred.tolist())
            prob_list.extend(prob.tolist())

    metric_log, results = single_model_gpu.get_eval_log(reset=True)
    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    prediction_file = os.path.join(cfg.output_dir, prefix, "eval_predictions.npy")
    np.save(prediction_file, pred_list)
    json.dump(prob_list, open(os.path.join(cfg.output_dir, prefix, "eval_probs.json"), "w"))

    return results


def load_and_cache_examples(cfg, tokenizer: PreTrainedTokenizer, _split="train"):
    if cfg.local_rank not in [-1, 0] and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if _split == "train":
        input_file = cfg.train_file
    elif _split == "dev":
        input_file = cfg.dev_file
    elif _split == "test":
        input_file = cfg.test_file
    else:
        raise RuntimeError(_split)

    examples, features, tensors = hydra.utils.call(cfg.read_tensor, file_path=input_file, tokenizer=tokenizer)

    if cfg.local_rank == 0 and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = TensorDataset(*tensors)

    return dataset, features


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        # dist.init_process_group(backend='nccl')
        deepspeed.init_distributed()
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)

    # Set seed
    set_seed(cfg)

    # Load pre-trained model and tokenizer
    if cfg.local_rank not in [-1, 0]:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if cfg.pretrain:
        pretrain_state_dict = torch.load(cfg.pretrain, map_location='cpu')
    else:
        pretrain_state_dict = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict)

    if cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # if cfg.local_rank == -1:  # For FullyShardedDDP, place the model on cpu first.
    #     model.to(cfg.device)

    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

    # Training
    if cfg.do_train:
        # TODO: Add option for continuously training from checkpoint.
        #  The operation should be introduced in ``train`` method since both the state dict
        #  of schedule and optimizer (and scaler, if any) should be loaded.
        # If output files already exists, assume to continue training from latest checkpoint (unless overwrite_output_dir is set)
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
        #     if len(checkpoints) > 0:
        #         checkpoint = checkpoints[-1]
        #         logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
        #         continue_from_global_step = int(checkpoint.split('-')[-1])
        #         model = model_class.from_pretrained(checkpoint)
        #         model.to(args.device)

        train_dataset, features = load_and_cache_examples(cfg, tokenizer, _split="train")
        global_step, tr_loss = train(cfg, train_dataset, features, model, tokenizer, continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Test
    results = {}
    if cfg.do_eval and cfg.local_rank in [-1, 0]:
        checkpoints = [cfg.output_dir]
        if cfg.save_best:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
            checkpoints = [cfg.prediction_cfg.best_checkpoint]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.eval_sub_path:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model.bin", recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            split = "dev"

            model = hydra.utils.call(cfg.model, checkpoint)
            model.to(device)

            if cfg.test_file:
                prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
                split = "test"

            result = evaluate(cfg, model, tokenizer, prefix=prefix, _split=split)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()

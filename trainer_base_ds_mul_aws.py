# coding=utf-8
#
# Copyright 2023 Nanyang Technological University Fangkai Jiao
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
import logging
import os
import sys
from typing import Dict, Union

import deepspeed
import hydra
import torch
import wandb
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)

from general_util.evaluator import evaluate_fn as evaluate
from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, unwrap_model, set_seed, note_best_checkpoint, load_and_cache_examples, set_seed_int

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True

GLOBAL_SEED = 1
GLOBAL_WORKER_ID = None


def get_zero_stage(cfg: DictConfig):
    if hasattr(cfg, "zero_optimization"):
        return int(getattr(cfg.zero_optimization, "stage", 0))
    return 0


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed_int(GLOBAL_SEED + worker_id)


def save_model(model: Union[deepspeed.DeepSpeedEngine, deepspeed.PipelineEngine],
               cfg: DictConfig, output_dir: str, tokenizer: PreTrainedTokenizer = None, state_dict: Dict = None):
    unwrapped_model = unwrap_model(model)
    model.save_checkpoint(output_dir)

    logger.info(f"Loading fp32 state dict from {output_dir}")
    zero_stage = get_zero_stage(cfg.ds_cfg)
    if zero_stage == 3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
    elif zero_stage == 2:
        state_dict = get_fp32_state_dict_from_zero_checkpoint(output_dir)
    else:
        state_dict = unwrapped_model.state_dict()

    if cfg.local_rank not in [-1, 0]:
        dist.barrier()

    if cfg.local_rank in [-1, 0]:
        # output_file = os.path.join(output_dir, "pytorch_model.bin")
        # print(f"Saving fp32 state dict to {output_file}")
        # torch.save(state_dict, output_file)
        unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)

        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)
        
        end_dir = output_dir.split("/")[-1]

        os.system(f"./s5cmd sync {output_dir}/ {cfg.aws_output_bucket}/{end_dir}/")

        if cfg.local_rank == 0:
            dist.barrier()


def forward_step(model, inputs: Dict[str, torch.Tensor]):
    outputs = model(**inputs)
    if isinstance(outputs, tuple):
        loss = outputs[0]
    else:
        loss = outputs["loss"]
    model.backward(loss)
    model.step()

    return loss.item(), outputs


def train(cfg, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        tb_helper = hydra.utils.instantiate(cfg.summary_helper) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_helper = None

    cfg.train_batch_size = cfg.per_gpu_train_batch_size

    if "_target_" in cfg.train_file:
        files = hydra.utils.instantiate(cfg.train_file)
    elif os.path.exists(cfg.train_file):
        files = [cfg.train_file]
    else:
        files = list(glob.glob(cfg.train_file))
    logger.info(files)

    if getattr(cfg, "total_dataset_len", -1) > 0:
        total_dataset_len = cfg.total_dataset_len
    else:
        total_dataset_len = 0
        for _file in tqdm(files, total=len(files)):
            sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            _train_sampler = RandomSampler(sub_train_dataset) if cfg.local_rank == -1 else DistributedSampler(sub_train_dataset)
            _train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
            _train_dataloader = DataLoader(dataset=sub_train_dataset,
                                           sampler=_train_sampler,
                                           batch_size=cfg.train_batch_size,
                                           collate_fn=_train_collator,
                                           num_workers=cfg.num_workers,
                                           pin_memory=True,
                                           prefetch_factor=cfg.prefetch_factor)
            total_dataset_len += len(_train_dataloader)
            del _train_dataloader
            del _train_collator
            del _train_sampler
            del sub_train_dataset

    if getattr(cfg, "do_preprocess", False):
        return

    if "extended_vocab" in cfg and cfg.extended_vocab:
        logger.info(f"Extended extra vocab size: {cfg.extended_vocab}")
        model.resize_token_embeddings(model.config.vocab_size + cfg.extended_vocab)

    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (total_dataset_len // cfg.gradient_accumulation_steps) + 1
    else:
        t_total = total_dataset_len // cfg.gradient_accumulation_steps * cfg.num_train_epochs

    num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    ds_config = cfg.ds_cfg
    ds_config.scheduler.params.total_num_steps = t_total
    ds_config.scheduler.params.warmup_num_steps = num_warmup_steps
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    # no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': cfg.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]
    torch.compile(model, mode="max-autotune")
    model, optimizer, _, scheduler = deepspeed.initialize(model=model,
                                                          model_parameters=model.parameters(),
                                                          config=ds_config)
    logger.info(optimizer.optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_dataset_len)
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                cfg.train_batch_size * cfg.gradient_accumulation_steps * (dist.get_world_size() if cfg.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
        model.load_checkpoint(cfg.resume)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        for _file in files:
            sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            sub_train_sampler = RandomSampler(sub_train_dataset) if cfg.local_rank == -1 else DistributedSampler(sub_train_dataset)
            sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
            sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                              sampler=sub_train_sampler,
                                              batch_size=cfg.train_batch_size,
                                              collate_fn=sub_train_collator,
                                              num_workers=cfg.num_workers,
                                              pin_memory=True,
                                              prefetch_factor=cfg.prefetch_factor,
                                              worker_init_fn=worker_init_fn)

            epoch_iterator = tqdm(sub_train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
            if cfg.local_rank != -1:
                sub_train_dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(epoch_iterator):
                # If training is continued from a checkpoint, fast forward
                # to the state of that checkpoint.
                if global_step < continue_from_global_step:
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        # scheduler.step()  # Update learning rate schedule  # Done by `load_checkpoint` of DS.
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
                    log_metrics = {}
                    if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                        log_metrics['lr'] = scheduler.get_lr()[0]
                        log_metrics['loss'] = (tr_loss - logging_loss) / cfg.logging_steps
                        logging_loss = tr_loss

                        if tb_helper:
                            log_metrics.update(tb_helper(last_batch=batch, last_outputs=outputs))

                    # Save model checkpoint
                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                        if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        save_model(model, cfg, output_dir, tokenizer)

                    # Evaluation
                    if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                        # state_dict = get_state_dict(model, cfg)

                        if cfg.ddp_eval or cfg.local_rank in [-1, 0]:
                            results = evaluate(cfg, model, tokenizer, prefix=str(global_step), _split="dev")

                            if cfg.local_rank in [-1, 0]:
                                for key, value in results.items():
                                    log_metrics[f"eval/{key}"] = value

                                sub_path = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                                flag = note_best_checkpoint(cfg, results, sub_path)
                                if cfg.save_best and flag:
                                    # save_model(model, cfg, cfg.output_dir, tokenizer, state_dict)
                                    # del state_dict
                                    save_model(model, cfg, cfg.output_dir, tokenizer)

                    if len(log_metrics) > 0 and cfg.local_rank in [-1, 0]:
                        wandb.log(log_metrics)

                    del batch
                    del log_metrics

                if 0 < cfg.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < cfg.max_steps < global_step:
                train_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            break

    return global_step, tr_loss / global_step


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != -1:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
    if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"]:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    if "WORLD_RANK" in os.environ and os.environ["WORLD_RANK"]:
        cfg.world_rank = int(os.environ["WORLD_RANK"])

    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed()
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

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

    from general_util.tokenization_utils import expand_special_tokenizer

    expand_special_tokenizer(tokenizer)

    try:
        model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict)
    except Exception as e:
        logger.warning(e)
        model = hydra.utils.call(cfg.model)

    if cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

        wandb.init(
            project="LLaMA-BiFLAN",
            name=f"{cfg.exp_name}-{dist.get_rank()}",
            notes=cfg.exp_notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric(cfg.prediction_cfg.metric, summary=("max" if cfg.prediction_cfg.measure > 0 else "min"))

    # Training
    if cfg.do_train:
        continue_from_global_step = 0  # If set to 0, start training from the beginning
        if os.path.exists(cfg.output_dir) and getattr(cfg, "resume", None):
            checkpoint = cfg.resume
            logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
            continue_from_global_step = int(checkpoint.split('-')[-1])

        global_step, tr_loss = train(cfg, model, tokenizer, continue_from_global_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Test
    results = {}
    if cfg.do_eval:
        if not cfg.ddp_eval and cfg.local_rank not in [-1, 0]:
            return results

        checkpoints = [cfg.output_dir]
        if cfg.save_best:
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
            checkpoints = [cfg.prediction_cfg.best_checkpoint]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        elif cfg.eval_sub_path:
            checkpoints = list(sorted(list(set(
                os.path.dirname(c) for c in
                glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model*.bin", recursive=True)
            ))))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            split = "dev"

            if "model_eval" in cfg:
                model = hydra.utils.call(cfg.model_eval, checkpoint)
            else:
                model = hydra.utils.call(cfg.model, checkpoint)

            model = deepspeed.init_inference(
                model,
                mp_size=cfg.world_size,
                dtype=torch.bfloat16,
                injection_policy=hydra.utils.instantiate(cfg.injection_policy) if "injection_policy" in cfg else None,
            )
            print(model.device)

            # if cfg.n_gpu == 1:
            #     model.to(cfg.device)
            # else:
            #     # For model parallel (of mT5)
            #     if getattr(cfg, "get_device_map", None):
            #         model.parallelize(hydra.utils.call(cfg.get_device_map))
            #     else:
            #         model.parallelize()

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
    print(sys.argv)
    main()

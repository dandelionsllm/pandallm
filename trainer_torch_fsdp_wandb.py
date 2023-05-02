# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import (DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, PreTrainedTokenizer)

from general_util.dist_utils import vanilla_torch_dist
from general_util.evaluator import evaluate_fn as evaluate
from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, unwrap_model, set_seed, note_best_checkpoint, \
    initialize_optimizer, \
    load_and_cache_examples, if_cancel_sync, initialize_lr_scheduler

"""
Requires torch >= 1.11.0 if you want to enable the torch vanilla FSDP instead of that of fairscale.
"""

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True


def save_model(model: Union[torch.nn.Module, FullyShardedDataParallel], cfg: DictConfig, output_dir: str,
               tokenizer: PreTrainedTokenizer = None):
    # Save model checkpoint.
    if cfg.local_rank != -1:
        state_dict = model.state_dict()
        if cfg.local_rank == 0:
            unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
    else:
        model.save_pretrained(output_dir)

    # Save tokenizer and training args.
    if cfg.local_rank in [-1, 0]:
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)


def forward_step(model, inputs: Dict[str, torch.Tensor], cfg, scaler, return_outputs: bool = False):
    if cfg.fp16:
        with torch.cuda.amp.autocast(dtype=(torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16)):
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    if isinstance(outputs, tuple):
        loss = outputs[0]
    else:
        loss = outputs["loss"]

    if cfg.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
    if cfg.gradient_accumulation_steps > 1:
        loss = loss / cfg.gradient_accumulation_steps

    if cfg.fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if return_outputs:
        return loss.item(), outputs

    return loss.item()


def train(cfg, train_dataset, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        tb_helper = hydra.utils.instantiate(
            cfg.summary_helper) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_helper = None

    # cfg.train_batch_size = cfg.per_gpu_train_batch_size * max(1, cfg.n_gpu)
    cfg.train_batch_size = cfg.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset) if cfg.local_rank == -1 else DistributedSampler(train_dataset)
    train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=train_sampler,
                                  batch_size=cfg.train_batch_size,
                                  collate_fn=train_collator,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  prefetch_factor=cfg.prefetch_factor)

    if "extended_vocab" in cfg and cfg.extended_vocab:
        logger.info(f"Extended extra vocab size: {cfg.extended_vocab}")
        model.resize_token_embeddings(model.config.vocab_size + cfg.extended_vocab)

    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (len(train_dataloader) // cfg.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // cfg.gradient_accumulation_steps * cfg.num_train_epochs

    num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    optimizer = scheduler = None
    # Prepare optimizer and schedule (linear warmup and decay)
    if cfg.local_rank == -1:
        optimizer = initialize_optimizer(cfg, model=model)
        scheduler = initialize_lr_scheduler(cfg, optimizer, num_warmup_steps, t_total)

    if cfg.fp16:
        if cfg.local_rank != -1:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            scaler = ShardedGradScaler()
        else:
            from torch.cuda.amp.grad_scaler import GradScaler

            scaler = GradScaler()
    else:
        scaler = None

    # Distributed training (should be after apex fp16 initialization)
    if cfg.local_rank != -1:
        if getattr(cfg, "fsdp_config", None):
            model = hydra.utils.instantiate(cfg.fsdp_config, model=model, device=cfg.device)
        elif getattr(cfg, "fairscale_config", None):
            model = hydra.utils.instantiate(cfg.fairscale_config, model=model, device=cfg.device)
        else:
            raise NotImplementedError
        optimizer = initialize_optimizer(cfg, model=model)
        scheduler = initialize_lr_scheduler(cfg, optimizer, num_warmup_steps, t_total)

    logger.info(optimizer)
    # logger.info(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                cfg.train_batch_size * cfg.gradient_accumulation_steps * (
                    dist.get_world_size() if cfg.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...",
                    continue_from_global_step)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0],
                              dynamic_ncols=True)
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

            last_outputs = None
            if if_cancel_sync(cfg, step):
                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                with model.no_sync():
                    loss = forward_step(model, batch, cfg, scaler)
            else:
                loss, last_outputs = forward_step(model, batch, cfg, scaler, return_outputs=True)

            tr_loss += loss
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.fp16:
                    scaler.unscale_(optimizer)

                if cfg.max_grad_norm and not ("optimizer" in cfg and cfg.optimizer and "lamb" in cfg.optimizer):
                    if hasattr(optimizer, "clip_grad_norm"):
                        optimizer.clip_grad_norm(cfg.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        model.clip_grad_norm_(cfg.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()  # Update learning rate schedule
                model.zero_grad(set_to_none=True)
                global_step += 1

                # Log metrics
                log_metrics = {}
                if cfg.local_rank in [-1, 0] and cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
                    log_metrics['lr'] = scheduler.get_lr()[0]
                    log_metrics['loss'] = (tr_loss - logging_loss) / cfg.logging_steps
                    logging_loss = tr_loss

                    if tb_helper:
                        log_metrics.update(tb_helper(last_batch=batch, last_outputs=last_outputs))

                # Save model checkpoint
                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                    if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(model, cfg, output_dir, tokenizer)

                # Evaluation
                if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    state_dict = model.state_dict()

                    if cfg.ddp_eval or cfg.local_rank in [-1, 0]:
                        results = evaluate(cfg, model, tokenizer, prefix=str(global_step), _split="dev")

                        if cfg.local_rank in [-1, 0]:
                            for key, value in results.items():
                                log_metrics[f"eval/{key}"] = value

                            sub_path = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                            flag = note_best_checkpoint(cfg, results, sub_path)
                            if cfg.save_best and flag:
                                if cfg.local_rank == 0:
                                    unwrap_model(model).save_pretrained(cfg.output_dir, state_dict=state_dict)
                                else:
                                    model.save_pretrained(cfg.output_dir)

                                tokenizer.save_pretrained(cfg.output_dir)
                                OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))
                                logger.info("Saving best model checkpoint to %s", cfg.output_dir)

                if len(log_metrics) > 0 and cfg.local_rank in [-1, 0]:
                    wandb.log(log_metrics)

                del batch
                del last_outputs

            if 0 < cfg.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if hasattr(cfg, "dist_init"):
        hydra.utils.instantiate(cfg.dist_init, cfg)
    else:
        vanilla_torch_dist(cfg)

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
    try:
        model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict)
    except Exception as e:
        logger.warning(e)
        model = hydra.utils.call(cfg.model)

    if cfg.local_rank == 0:
        dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if cfg.local_rank == -1:  # For FullyShardedDDP, place the model on cpu first.
        if cfg.n_gpu in [0, 1] or cfg.no_cuda:
            model.to(cfg.device)
        else:
            # For model parallel (of mT5)
            logger.info(f"Model Parallel initialization.")
            model.parallelize(hydra.utils.call(cfg.get_device_map))

    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

        wandb.init(
            project="code-language-reasoning",
            name=cfg.exp_name,
            notes=cfg.exp_notes,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.define_metric(cfg.prediction_cfg.metric, summary=("max" if cfg.prediction_cfg.measure > 0 else "min"))

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

        train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train")

        if getattr(cfg, "do_preprocess", False):
            return

        global_step, tr_loss = train(cfg, train_dataset, model, tokenizer, continue_from_global_step)
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

            if "model_eval" in cfg:
                model = hydra.utils.call(cfg.model_eval, checkpoint)
            else:
                model = hydra.utils.call(cfg.model, checkpoint)
            if cfg.n_gpu == 1:
                model.to(cfg.device)
            else:
                # For model parallel (of mT5)
                model.parallelize(hydra.utils.call(cfg.get_device_map))

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

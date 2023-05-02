import json
import os
import inspect

import hydra
import torch
from torch import distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.generation.configuration_utils import GenerationConfig

from general_util.logger import get_child_logger
from general_util.training_utils import batch_to_device, load_and_cache_examples, unwrap_model

logger = get_child_logger(__name__)


def evaluate(cfg: DictConfig, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)

    output_dir = getattr(cfg, "predict_dir", cfg.output_dir)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(output_dir, prefix)):
        os.makedirs(os.path.join(output_dir, prefix))

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    if cfg.ddp_eval and cfg.local_rank != -1:
        eval_sampler = DistributedSampler(dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly

    eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator)

    post_processor = hydra.utils.instantiate(cfg.post_process) if "post_process" in cfg and cfg.post_process else None

    single_model_gpu = unwrap_model(model)
    if hasattr(single_model_gpu, "get_eval_log"):
        single_model_gpu.get_eval_log(reset=True)
    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()
    pred_list = []
    indices_list = []

    torch.cuda.empty_cache()

    # prediction_state = _split == "test" and getattr(cfg, "generator", False)
    prediction_state = getattr(cfg, "generator", False)
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
        if "meta_data" in batch:
            meta_data = batch.pop("meta_data")
        else:
            meta_data = []
        if "index" in batch:
            indices_list.extend(batch.pop("index").tolist())

        batch = batch_to_device(batch, cfg.device)
        if cfg.fp16:
            with torch.cuda.amp.autocast(
                    dtype=(torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16)):
                with torch.no_grad():
                    if not prediction_state:
                        outputs = model(**batch)
                        probs = outputs["logits"].softmax(dim=-1).detach().float().cpu()

                        _, pred = probs.max(dim=-1)
                        pred_list.extend(pred.tolist())
                    else:
                        outputs = model(**batch, disable_decoder=True)

                        if getattr(cfg, "output_scores", False):
                            decoding_outputs = model.generate(**batch, max_length=cfg.max_output_length,
                                                              num_beams=cfg.num_beams,
                                                              num_return_sequences=cfg.num_return_sequences,
                                                              output_scores=getattr(cfg, "output_scores", False),
                                                              return_dict_in_generate=True)

                            generated_seq = tokenizer.batch_decode(decoding_outputs["sequences"],
                                                                   skip_special_tokens=True)
                            outputs["generated_seq"] = generated_seq
                            outputs["sequences_scores"] = decoding_outputs["sequences_scores"]
                        else:
                            decoding_outputs = model.generate(**batch, max_length=cfg.max_output_length,
                                                              num_beams=cfg.num_beams,
                                                              num_return_sequences=cfg.num_return_sequences)
                            decoding_outputs = tokenizer.batch_decode(decoding_outputs, skip_special_tokens=True)
                            outputs["generated_seq"] = decoding_outputs
        else:
            with torch.no_grad():
                if not prediction_state:
                    outputs = model(**batch)
                    probs = outputs["logits"].softmax(dim=-1).detach().float().cpu()

                    _, pred = probs.max(dim=-1)
                    pred_list.extend(pred.tolist())
                else:
                    outputs = model(**batch, disable_decoder=True)

                    if getattr(cfg, "output_scores", False):
                        decoding_outputs = model.generate(**batch, max_length=cfg.max_output_length,
                                                          num_beams=cfg.num_beams,
                                                          num_return_sequences=cfg.num_return_sequences,
                                                          output_scores=getattr(cfg, "output_scores", False),
                                                          return_dict_in_generate=True)
                        generated_seq = tokenizer.batch_decode(decoding_outputs["sequences"], skip_special_tokens=True)
                        outputs["generated_seq"] = generated_seq
                        outputs["sequences_scores"] = decoding_outputs["sequences_scores"]
                    else:
                        decoding_outputs = model.generate(**batch, max_length=cfg.max_output_length,
                                                          num_beams=cfg.num_beams,
                                                          num_return_sequences=cfg.num_return_sequences)
                        decoding_outputs = tokenizer.batch_decode(decoding_outputs, skip_special_tokens=True)
                        outputs["generated_seq"] = decoding_outputs

        if post_processor is not None:
            if any(hasattr(post_processor, tmp) for tmp in ["gather", "gather_object"]):
                kwargs = {
                    "ddp": cfg.ddp_eval and cfg.local_rank != -1
                }
            else:
                kwargs = {}
            post_processor(meta_data, outputs, **kwargs)

    if hasattr(single_model_gpu, "get_eval_log"):
        metric_log, results = single_model_gpu.get_eval_log(reset=True, ddp=(cfg.ddp_eval and cfg.local_rank != -1),
                                                            device=cfg.device)
    else:
        results = {}
        metric_log = ""

    if post_processor is not None:
        post_results, post_predictions = post_processor.get_results()
        results.update(post_results)
        metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])
        predictions = post_predictions
    else:
        predictions = pred_list

    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    if cfg.local_rank == -1:
        prediction_file = os.path.join(output_dir, prefix, "eval_predictions.json")
    else:
        prediction_file = os.path.join(output_dir, prefix, f"eval_predictions_rank{cfg.local_rank}.json")
    json.dump(predictions, open(prediction_file, "w"), ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()

    return results


def evaluate_fn(cfg: DictConfig, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    # logger = get_child_logger(__name__)

    dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)

    output_dir = getattr(cfg, "predict_dir", cfg.output_dir)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(output_dir, prefix)):
        os.makedirs(os.path.join(output_dir, prefix))

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    if cfg.ddp_eval and cfg.local_rank != -1:
        eval_sampler = DistributedSampler(dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly

    eval_collator_cfg = getattr(cfg, f"{_split}_collator", None)
    if eval_collator_cfg is not None:
        eval_collator = hydra.utils.instantiate(eval_collator_cfg)
    else:
        eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 prefetch_factor=cfg.prefetch_factor)

    post_processor = hydra.utils.instantiate(cfg.post_process) if "post_process" in cfg and cfg.post_process else None

    single_model_gpu = unwrap_model(model)
    if hasattr(single_model_gpu, "get_eval_log"):
        single_model_gpu.get_eval_log(reset=True)
    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()
    pred_list = []
    indices_list = []

    eval_forward_fn = hydra.utils.instantiate(cfg.eval_forward_fn, cfg, model, tokenizer)

    # logger.info(f"If model is an instance of FSDP: {isinstance(model, FSDP)}")
    # logger.info(f"If encoder is an instance of FSDP: {isinstance(model.module.encoder, FSDP)}")

    torch.cuda.empty_cache()
    pbar = tqdm(eval_dataloader, desc="Evaluating", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
    for itr, batch in enumerate(pbar):
        if "meta_data" in batch:
            meta_data = batch.pop("meta_data")
        else:
            meta_data = []
        if "index" in batch:
            indices_list.extend(batch.pop("index").tolist())

        batch = batch_to_device(batch, cfg.device)
        auto_cast_param_dict = {
            "enabled": cfg.fp16,
            "dtype": torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16
        }
        with torch.cuda.amp.autocast(**auto_cast_param_dict):
            with torch.no_grad():
                outputs, pred_res = eval_forward_fn(batch)
                # pbar.set_description(f'Loss = {outputs.loss.item()}')
                # if outputs.loss.isnan():
                #     raise ValueError(f'NaN raised at itr = {itr}! Batch = {batch}')
        pred_list.extend(pred_res)

        if post_processor is not None:
            if any(hasattr(post_processor, tmp) for tmp in ["gather", "gather_object"]):
                kwargs = {
                    "ddp": cfg.ddp_eval and cfg.local_rank != -1
                }
            else:
                kwargs = {}
            post_processor(meta_data, outputs, **kwargs)

    if hasattr(single_model_gpu, "get_eval_log"):
        metric_log, results = single_model_gpu.get_eval_log(reset=True, ddp=(cfg.ddp_eval and cfg.local_rank != -1),
                                                            device=cfg.device)
    else:
        results = {}
        metric_log = ""

    if post_processor is not None:
        sig = inspect.signature(post_processor.get_results)
        post_kwargs = {}
        # print(sig.parameters)
        # print(sig.parameters.keys())
        if "output_dir" in list(sig.parameters.keys()):
            post_kwargs["output_dir"] = os.path.join(output_dir, prefix)

        post_results, post_predictions = post_processor.get_results(**post_kwargs)
        results.update(post_results)
        metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])
        predictions = post_predictions
    else:
        predictions = pred_list

    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    if len(predictions) > 0:
        if cfg.local_rank == -1:
            prediction_file = os.path.join(output_dir, prefix, "eval_predictions.json")
        else:
            prediction_file = os.path.join(output_dir, prefix, f"eval_predictions_rank{cfg.local_rank}.json")
        json.dump(predictions, open(prediction_file, "w"), ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()

    return results


def build_dataloader(dataset, cfg):
    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    if cfg.ddp_eval and cfg.local_rank != -1:
        eval_sampler = DistributedSampler(dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly

    eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 prefetch_factor=cfg.prefetch_factor)

    return eval_dataloader


def retriever_inference_fn(cfg: DictConfig, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, prefix="", _split="dev"):
    # dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)
    # Just a hack here. We use the training set to indicate the document dataset while the dev or test dataset as the query dataset.
    doc_dataset = load_and_cache_examples(cfg, tokenizer, _split="train")
    que_dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)

    output_dir = getattr(cfg, "predict_dir", cfg.output_dir)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(output_dir, prefix)):
        os.makedirs(os.path.join(output_dir, prefix))

    doc_dataloader = build_dataloader(doc_dataset, cfg)
    que_dataloader = build_dataloader(que_dataset, cfg)

    post_processor = hydra.utils.instantiate(cfg.post_process) if "post_process" in cfg and cfg.post_process else None

    single_model_gpu = unwrap_model(model)
    if hasattr(single_model_gpu, "get_eval_log"):
        single_model_gpu.get_eval_log(reset=True)
    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Building index {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(doc_dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()
    doc_pred_list = []
    doc_indices_list = []

    # eval_forward_fn = hydra.utils.instantiate(cfg.eval_forward_fn, cfg, model, tokenizer)

    torch.cuda.empty_cache()
    for batch in tqdm(doc_dataloader, desc="Building", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
        if "meta_data" in batch:
            meta_data = batch.pop("meta_data")
        else:
            meta_data = []
        if "index" in batch:
            doc_indices_list.extend(batch.pop("index").tolist())

        batch = batch_to_device(batch, cfg.device)
        auto_cast_param_dict = {
            "enabled": cfg.fp16,
            "dtype": torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16
        }
        with torch.cuda.amp.autocast(**auto_cast_param_dict):
            with torch.no_grad():
                model.encode_index(**batch)

        # pred_list.extend(pred_res)

        # if post_processor is not None:
        #     if any(hasattr(post_processor, tmp) for tmp in ["gather", "gather_object"]):
        #         kwargs = {
        #             "ddp": cfg.ddp_eval and cfg.local_rank != -1
        #         }
        #     else:
        #         kwargs = {}
        #     post_processor(meta_data, outputs, **kwargs)

    que_indices_list = []
    que_pred_list = []

    torch.cuda.empty_cache()
    for batch in tqdm(que_dataloader, desc="Building", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True):
        if "meta_data" in batch:
            meta_data = batch.pop("meta_data")
        else:
            meta_data = []
        if "index" in batch:
            que_indices_list.extend(batch.pop("index").tolist())

        batch = batch_to_device(batch, cfg.device)
        auto_cast_param_dict = {
            "enabled": cfg.fp16,
            "dtype": torch.bfloat16 if getattr(cfg, "fp16_bfloat16", False) else torch.float16
        }
        with torch.cuda.amp.autocast(**auto_cast_param_dict):
            with torch.no_grad():
                scores = model.search(**batch).cpu()
                que_pred_list.append(scores)

    if hasattr(single_model_gpu, "get_eval_log"):
        metric_log, results = single_model_gpu.get_eval_log(reset=True, ddp=(cfg.ddp_eval and cfg.local_rank != -1),
                                                            device=cfg.device)
    else:
        results = {}
        metric_log = ""

    if post_processor is not None:
        sig = inspect.signature(post_processor.get_results)
        post_kwargs = {}
        # print(sig.parameters)
        # print(sig.parameters.keys())
        if "output_dir" in list(sig.parameters.keys()):
            post_kwargs["output_dir"] = os.path.join(output_dir, prefix)

        post_results, post_predictions = post_processor.get_results(**post_kwargs)
        results.update(post_results)
        metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])
        predictions = post_predictions
    else:
        predictions = torch.cat(que_pred_list, dim=0)

    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    if len(predictions) > 0:
        if cfg.local_rank == -1:
            prediction_file = os.path.join(output_dir, prefix, "eval_predictions.json")
        else:
            prediction_file = os.path.join(output_dir, prefix, f"eval_predictions_rank{cfg.local_rank}.json")
        json.dump(predictions, open(prediction_file, "w"), ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()

    return results


class DefaultForwardFn:
    def __init__(self, cfg: DictConfig, model: torch.nn.Module, tokenizer: PreTrainedTokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, batch):
        outputs = self.model(**batch)
        return outputs, []


class DiscriminatorForwardFn:
    def __init__(self, cfg: DictConfig, model: torch.nn.Module, tokenizer: PreTrainedTokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, batch):
        outputs = self.model(**batch)
        probs = outputs["logits"].softmax(dim=-1).detach().float().cpu()

        _, pred = probs.max(dim=-1)
        return outputs, pred.tolist()


class GeneratorForwardFn:
    def __init__(self, cfg: DictConfig, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, generation_config: GenerationConfig,
                 skip_special_tokens: bool = True):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, batch):
        if "labels" in batch:  # Kept as the `decoder_input_ids`. Should be removed during auto-regressive inference.
            batch.pop("labels")

        outputs = {}
        decoding_outputs = self.model.generate(**batch, generation_config=self.generation_config)

        if self.generation_config.output_scores:
            generated_seq = self.tokenizer.batch_decode(decoding_outputs["sequences"], skip_special_tokens=self.skip_special_tokens)
            outputs["generated_seq"] = generated_seq
            outputs["sequences_scores"] = decoding_outputs["sequences_scores"]
        else:
            generated_seq = self.tokenizer.batch_decode(decoding_outputs, skip_special_tokens=self.skip_special_tokens)
            outputs["generated_seq"] = generated_seq

        return outputs, []


class GeneratorCLSForwardFn(GeneratorForwardFn):
    def __call__(self, batch):
        # FIXME: Currently, we have to perform an extra forward to avoid a strange issue caused by FSDP,
        #  no matter if we really need the outputs from the encoder.
        #  Anyway, if the model is not warpped by FSDP, this step can be omitted.
        #  For the details, please refer to https://github.com/pytorch/pytorch/issues/82461
        outputs = self.model(**batch, disable_decoder=True)

        _generate_outputs, res = super(GeneratorCLSForwardFn, self).__call__(batch)
        for key, val in _generate_outputs.items():
            outputs[key] = val
        return outputs, res

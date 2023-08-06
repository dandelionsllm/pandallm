import os
import random
from typing import Union, Tuple, Optional, Dict, Callable

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer
from models.mpt.modeling_mpt import attn_bias_shape, build_attn_bias, MPTConfig

logger = get_child_logger("FLAN")


class FLANDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        self.data = torch.load(file_path, map_location="cpu")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class WikiPathDatasetV5WFlan(Dataset):
    def __init__(self, raw_data: Union[Tuple, DictConfig], flan_file: str, file_path: str, tokenizer: PreTrainedTokenizer):
        # print(type(raw_data))
        if isinstance(raw_data, DictConfig):
            raw_data = hydra.utils.instantiate(raw_data, file_path=file_path, tokenizer=tokenizer)

        self.examples = raw_data[0]
        self.flan_data = torch.load(flan_file, map_location="cpu")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if index >= len(self.flan_data):
            flan = random.choice(self.flan_data)
        else:
            flan = self.flan_data[index]
        return {
            "example": example,
            "flan": flan,
            "index": index,
        }


class FlanCollectionGroupDataset(Dataset):
    def __init__(self, file_path: str, tokenizer=None):
        super().__init__()
        logger.info(f"Loading FLAN data from {file_path}...")
        data = torch.load(file_path, map_location="cpu")
        self.data = []
        cnt = 0
        for item in data:
            if item["inputs"].strip() == "":
                continue
            if item["targets"].strip() == "":
                cnt += 1
                continue
            self.data.append(item)
        logger.info(f"Removed {cnt} empty examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


def vanilla_seq2seq_convertor(examples, tokenizer: PreTrainedTokenizer, max_seq_length, decoder_only: bool = False,
                              padding: str = "max_length"):
    inputs = []
    outputs = []
    for exp in examples:
        inputs.append(exp["inputs"])
        if decoder_only:
            outputs.append(exp["inputs"] + " " + exp["targets"] + tokenizer.eos_token)
        else:
            outputs.append(exp["targets"])

    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding=padding,
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding=padding, truncation=True, return_tensors="pt")
        new_input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        input_lens = input_lens - input_lens.eq(new_input_lens).to(input_lens.dtype) * (input_lens // 2)
        input_lens = input_lens.to(torch.long)
        if tokenizer.padding_side == "left":
            input_lens = model_inputs["input_ids"].eq(tokenizer.pad_token_id).to(torch.long).sum(dim=1) + input_lens
        model_inputs["input_lens"] = input_lens

    return model_inputs


class FlanCollatorOverCollator:
    def __init__(self, collator, tokenizer: str, max_seq_length: int, decoder_only: bool = False, padding="max_length",
                 pp_inputs_processor: Callable = None, **kwargs):
        self.collator = collator
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only
        self.padding = padding
        self.pp_inputs_processor = pp_inputs_processor

    def __call__(self, batch):
        flan_batch = []
        for item in batch:
            flan_batch.append(item.pop("flan"))

        if self.collator is not None:
            model_inputs = self.collator(batch)
            flan_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only, self.padding)
            for k, v in flan_inputs.items():
                model_inputs[f"flan_{k}"] = v
        else:
            model_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only, self.padding)

        if self.pp_inputs_processor is not None:
            return self.pp_inputs_processor(model_inputs, self.tokenizer)

        return model_inputs


class CombineCollator:
    def __init__(self, tokenizer: str, max_seq_length: int, decoder_only: bool = False, padding="max_length",
                 pp_inputs_processor: Callable = None, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only
        self.padding = padding
        self.pp_inputs_processor = pp_inputs_processor

    def __call__(self, batch):
        texts = [b.pop("flan") for b in batch]
        texts += [b.pop("extra") for b in batch]
        model_inputs = vanilla_seq2seq_convertor(texts, self.tokenizer, self.max_seq_length, self.decoder_only, self.padding)

        if self.pp_inputs_processor is not None:
            return self.pp_inputs_processor(model_inputs, self.tokenizer)

        return model_inputs

import random
import json
import torch
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer
from typing import Callable
import hydra
import pandas as pd
from omegaconf import DictConfig
import json
from typing import Union, Tuple
import os
import gzip
import warnings
from datasets import load_dataset

warnings.simplefilter('ignore', FutureWarning)

logger = get_child_logger("MMLU")


class MMLUDataset(Dataset):
    def __init__(self, file_path: str, data_dir=None, tokenizer=None):
        super().__init__()
        self.data = []
        df = pd.read_csv(file_path, index_col=0)
        for i in range(df.shape[0]):
            self.data.append(df.iloc[i, :].to_dict())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


class HumanEvalDataset(Dataset):
    def __init__(self, file_path: str, data_dir=None, tokenizer=None):
        super().__init__()
        self.data = []
        data = load_dataset("openai_humaneval")['test']
        for sample in data:
            self.data.append({
                'inputs': sample['prompt'],
                'targets': sample['canonical_solution'],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


class GSM8KDataset(Dataset):
    def __init__(self, file_path: str, data_dir=None, tokenizer=None):
        super().__init__()
        self.data = []
        data_main = load_dataset("gsm8k", 'main')['test']
        data_sac = load_dataset('gsm8k', 'socratic')['test']
        for sample in data_main:
            self.data.append({
                'inputs': sample['question'],
                'targets': sample['answer'],
            })
        for sample in data_sac:
            self.data.append({
                'inputs': sample['question'],
                'targets': sample['answer'],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


class NaturalQuestionsDataset(Dataset):
    # https://ai.google.com/research/NaturalQuestions/download
    def __init__(self, file_path: str, data_dir=None, tokenizer=None):
        super().__init__()
        # Open the JSON file for reading
        data = []
        with open(file_path, 'r') as file:
            # Iterate over the lines in the file
            for line in file:
                # Load the JSON data from each line
                data.append(json.loads(line))

        self.data = []
        for sample in data:
            answer = self.get_data_answer(sample)
            if answer is not None:
                self.data.append({
                    'inputs': sample['question_text'],
                    'targets': answer,
                })
        print(f'Natural question dataset constructed with {self.__len__()} data.')

    def get_data_answer(self, sample):
        answer = None
        for annotation in sample['annotations']:
            if len(annotation['short_answers']) > 0:
                short_answer = annotation['short_answers'][0]
                answer = [sample['document_tokens'][i]
                          for i in range(short_answer['start_token'], short_answer['end_token'])]
                answer = ' '.join(answer)
                pattern = r"<.*?>"
                answer = re.sub(pattern, "", answer)

                return answer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


class TruthfulQADataset(Dataset):
    def __init__(self, file_path: str, data_dir=None, tokenizer=None):
        super().__init__()
        self.data = []
        data_mul = load_dataset('truthful_qa', 'multiple_choice')['validation']
        data_gen = load_dataset('truthful_qa', 'generation')['validation']
        self.collect_data(data_mul, mode='multiple_choice')
        self.collect_data(data_gen, mode='generation')

    def collect_data(self, data, mode):
        if mode == 'multiple_choice':
            for sample in data:
                inputs = [sample['question']]
                choices = [f'{num + 1}. {choice}' for num, choice in enumerate(sample['mc1_targets']['choices'])]
                inputs += choices
                inputs = '\n'.join(inputs)
                best_choice = [f'{num + 1}.' for num, label in enumerate(sample['mc1_targets']['labels'])
                                if label == 1][0]
                self.data.append({
                    'inputs': inputs,
                    'targets': f'The best choices is {best_choice}.',
                })

        elif mode == 'generation':
            for sample in data:
                self.data.append({
                    'inputs': sample['question'],
                    'targets': sample['best_answer'],
                })
        print(f'{mode} data loaded successfully ...')



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "flan": self.data[index],
        }


def vanilla_seq2seq_convertor(examples, tokenizer, max_seq_length, decoder_only: bool = False):
    inputs = []
    outputs = []
    for exp in examples:
        inputs.append(exp["inputs"])
        if decoder_only:
            outputs.append(exp["inputs"] + " " + exp["targets"])
        else:
            outputs.append(exp["targets"])

    model_inputs = tokenizer(inputs, text_target=outputs, max_length=max_seq_length, padding="longest",
                             truncation=True, return_tensors="pt")
    if decoder_only:
        input_lens = model_inputs["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
        model_inputs = tokenizer(outputs, max_length=max_seq_length, padding="longest",
                                 truncation=True, return_tensors="pt")
        model_inputs["input_lens"] = input_lens

    return model_inputs


class GeneralCollatorOverCollator:
    def __init__(self, collator, tokenizer: str, max_seq_length: int, decoder_only: bool = False):
        self.collator = collator
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only

    def __call__(self, batch):
        flan_batch = []
        for item in batch:
            flan_batch.append(item.pop("flan"))

        if self.collator is not None:
            model_inputs = self.collator(batch)
            flan_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only)
            for k, v in flan_inputs.items():
                model_inputs[f"flan_{k}"] = v
        else:
            model_inputs = vanilla_seq2seq_convertor(flan_batch, self.tokenizer, self.max_seq_length, self.decoder_only)

        return model_inputs

import random
from typing import List

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from general_util.tokenization_utils import expand_special_tokenizer


_default_instruct = "Answer the following question with the given context:"

_rank2option = ["A", "B", "C", "D", "E"]


def _format_option_list(option_list: List[str]) -> str:
    res = ""
    for op_id, op in enumerate(option_list):
        res += f"{_rank2option[op_id]}: {op}\n"
    return res


class ReClorExemplarGenerator:
    def __init__(self, file_path, read_func, shot: int = 0, random_sampling: bool = False, instruct: str = _default_instruct):
        self.shot = shot
        self.random = random_sampling
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        index = list(range(len(all_context)))

        if self.random:
            random.shuffle(index)

        prompts = []
        for i in index[:self.shot]:
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\nThe answer is {_rank2option[all_label[i]]}"
            prompts.append(prompt)

        self.prompt = instruct + "\n\n" + "\n\n".join(prompts)
        self.indices = index[:self.shot]

    def __call__(self):
        return self.prompt, self.indices


class ReClorGenerativeDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is"):
        self.prompt_generator = prompt_generator
        # read_func = ReClorReader()
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            prompt = f"Context:\n{all_context[i]}\n\nQuestion:\n{all_question[i]}\n\nOptions:\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n" + suffix
            self.inputs.append(prompt)
            self.indices.append(i)
            self.labels.append(_rank2option[all_label[i]])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        prompt, prompt_indices = self.prompt_generator()
        return {
            "input": prompt + "\n\n" + self.inputs[index],
            "index": self.indices[index],
            "prompt_index": ",".join(map(str, prompt_indices)),
            "label": self.labels[index],
        }


class ReClorSeq2SeqMCQADataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func):
        super().__init__()
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        self.contexts = all_context
        self.questions = all_question
        self.option_list = all_option_list
        self.labels = all_label

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        return {
            "index": index,
            "context": self.contexts[index],
            "question": self.questions[index],
            "options": self.option_list[index],
            "label": self.labels[index],
        }


class ReClorSeq2SeqMCQACollator:
    def __init__(self, tokenizer: str, max_seq_length: int, decoder_only: bool = False):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length
        self.decoder_only = decoder_only

    def __call__(self, batch):
        inputs_a = []
        inputs_b = []
        batch_size = len(batch)
        labels = []
        indices = []
        for b in batch:
            op_num = len(b["options"])
            inputs_a.extend([b["context"] + b["question"]] * op_num)
            if self.decoder_only:
                inputs_b.extend(list(map(lambda x: b["context"] + b["question"] + x, b["options"])))
            else:
                inputs_b.extend(b["options"])
            labels.append(b["label"])
            indices.append(b["index"])

        op_num = len(inputs_a) // batch_size

        model_inputs = self.tokenizer(inputs_a, text_target=inputs_b,
                                      padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.decoder_only:
            input_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1)
            model_inputs = self.tokenizer(inputs_b, padding="longest", truncation=True, max_length=self.max_seq_length,
                                          return_tensors="pt")
            model_inputs["input_lens"] = input_lens

        model_inputs["input_ids"] = model_inputs["input_ids"].reshape(batch_size, op_num, -1)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].reshape(batch_size, op_num, -1)

        if not self.decoder_only:
            model_inputs["decoder_input_ids"] = model_inputs["labels"].reshape(batch_size, op_num, -1)

        model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        model_inputs["meta_data"] = {"index": indices}
        return model_inputs


class ReClorGenerativeCollator:
    def __init__(self, tokenizer: str, max_seq_length: int):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        batch = default_collate(batch)
        inputs = batch.pop("input")
        model_inputs = self.tokenizer(inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        # remove `eos_token_id` from `input_ids`
        # eos_mask = model_inputs["input_ids"].eq(self.tokenizer.eos_token_id)
        # model_inputs["input_ids"][eos_mask] = self.tokenizer.pad_token_id
        # model_inputs["attention_mask"][eos_mask] = 0

        model_inputs["meta_data"] = batch
        model_inputs["meta_data"]["input"] = inputs
        return model_inputs

import json
import random
from typing import List

import torch
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer, AutoTokenizer
from general_util.tokenization_utils import expand_special_tokenizer
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

_default_instruct = "Answer the following question with the given context:"

_rank2option = ["A", "B", "C", "D", "E"]

instruction_list = {
    "vicuna_style": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                    "\n\n### Instruction: ",
}


def compose_flat_prompt_input(context, question, option_list) -> str:
    context = "Context:\n" + context
    question = "Question:\n" + question
    option_list = option_list
    options = "Options:\n" + "\n".join([chr(ord('A') + i) + ": " + option for i, option in enumerate(option_list)])
    return "\n\n".join([context, question, options])


def read_cot_prompts(data_file, prediction_file, sample_num: int = 5, random_sample: bool = False, self_consistency: bool = True):
    data = json.load(open(data_file, 'r'))
    predictions = json.load(open(prediction_file, 'r'))
    prompts = []
    for item, pred in zip(data, predictions):
        assert item["id_string"] == pred["id"]
        if not self_consistency or item["label"] == int(ord(pred["pred"]) - ord("A")):
            inputs = compose_flat_prompt_input(item["context"], item["question"], item["answers"])
            response = pred["response"]
            prompts.append(inputs + "\n\n" + response)
    if random_sample:
        random.shuffle(prompts)
    logger.info(f"Filtered {len(prompts)} prompts.")

    def _callable():
        return prompts[:sample_num]

    return _callable


def read_cot_prompts_logiqa_v2(data_file, prediction_file, sample_num: int = 5, random_sample: bool = False, self_consistency: bool = True,
                               instruct: str = _default_instruct):
    data = open(data_file, 'r').readlines()
    predictions = json.load(open(prediction_file, 'r'))
    prompts = []
    for line, pred in zip(data, predictions):
        item = json.loads(line)
        assert item["id"] == pred["id"]
        if not self_consistency or pred["pred"].strip() and item["answer"] == int(ord(pred["pred"].strip()) - ord("A")):
            inputs = compose_flat_prompt_input(item["text"], item["question"], item["options"])
            response = pred["response"]
            prompts.append(inputs + "\n\n" + response)
    if random_sample:
        random.shuffle(prompts)
    logger.info(f"Filtered {len(prompts)} prompts.")

    def _callable():
        return instruct + "\n\n" + "\n\n".join(prompts[:sample_num]), list(range(sample_num))

    return _callable


def read_raw_prompts_logiqa_v2(data_file, sample_num: int = 5, random_sample: bool = False):
    data = open(data_file, 'r').readlines()
    prompts = []
    for line in data:
        item = json.loads(line)
        inputs = compose_flat_prompt_input(item["text"], item["question"], item["options"])
        inputs = inputs + "\n\n" + "The answer is " + chr(ord('A') + item["answer"]) + "."
        prompts.append(inputs)
    if random_sample:
        random.shuffle(prompts)
    return prompts[:sample_num]


def read_cot_prompts_logiqa_v2_category(data_file, prediction_file, test_file, sample_num: int = 5, random_sample: bool = False,
                                        cate_overlap_num: int = 2):
    data = list(map(json.loads, open(data_file, 'r').readlines()))
    predictions = json.load(open(prediction_file, 'r'))
    test_data = list(map(json.loads, open(test_file, 'r').readlines()))
    all_prompts = []
    for item, pred in zip(data, predictions):
        assert item["id"] == pred["id"]
        if pred["pred"].strip() and item["answer"] == int(ord(pred["pred"].strip()) - ord("A")):
            item["response"] = pred["response"]
            all_prompts.append(item)

    logger.info(f"Filtered {len(all_prompts)} prompts.")
    item_prompts = []
    less = 0
    for item_id, item in enumerate(test_data):
        item_reason_types = set([r for r, f in item["type"].items() if f])
        tmp = []
        for prompt in all_prompts:
            prompt_reason_types = set([r for r, f in prompt["type"].items() if f])
            if len(item_reason_types & prompt_reason_types) >= cate_overlap_num:
                tmp.append(prompt)
        if random_sample:
            random.shuffle(tmp)
        if len(tmp) < sample_num:
            less += 1
        if len(tmp) == 0:
            if random_sample:
                tmp = random.sample(all_prompts, sample_num)
            else:
                tmp = all_prompts[:sample_num]
        tmp = [compose_flat_prompt_input(x["text"], x["question"], x["options"]) + "\n\n" + x["response"]
               for x in tmp[:sample_num]]
        item_prompts.append(tmp)

    logger.info(f"Less than {sample_num} prompts: {less}")
    return item_prompts


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


class ReClorExemplarGeneratorZh:
    def __init__(self, file_path, read_func, shot: int = 0, random_sampling: bool = False, instruct: str = _default_instruct):
        self.shot = shot
        self.random = random_sampling
        all_context, all_question, all_option_list, all_label = read_func(file_path)
        index = list(range(len(all_context)))

        if self.random:
            random.shuffle(index)

        prompts = []
        for i in index[:self.shot]:
            prompt = f"文章：\n{all_context[i]}\n\n问题：\n{all_question[i]}\n\n选项：\n{_format_option_list(all_option_list[i])}" \
                     f"\n\n答案是 {_rank2option[all_label[i]]}"
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


class ReClorGenerativeDatasetZh(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, prompt_generator, suffix: str = "The answer is"):
        self.prompt_generator = prompt_generator
        # read_func = ReClorReader()
        all_context, all_question, all_option_list, all_label = read_func(file_path)

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(all_context)):
            prompt = f"{all_context[i]}\n\n问题：\n{all_question[i]}\n\n选项：\n{_format_option_list(all_option_list[i])}" \
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
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer)
        expand_special_tokenizer(self.tokenizer)
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        batch = default_collate(batch)
        inputs = batch.pop("input")
        model_inputs = self.tokenizer(inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        # print(model_inputs.keys())
        if "token_type_ids" in model_inputs:
            model_inputs.pop("token_type_ids")
        model_inputs["meta_data"] = batch
        model_inputs["meta_data"]["input"] = inputs
        return model_inputs


class ReClorChatDataset(Dataset):
    """
    For post-processing by chat.
    The input file should be the output of `post_processors.reclor.GeneratorPredictor`.
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, instruction: str, suffix: str):
        self.data = json.load(open(file_path, "r"))
        self.instruction = instruction
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Should note that in chat format, instruction follows original output."""
        inputs = self.data[index]["output"] + self.instruction + self.suffix
        return {
            "input": inputs,
            "index": index,
            "prompt_index": "0",
            "label": self.data[index]["label"],
        }

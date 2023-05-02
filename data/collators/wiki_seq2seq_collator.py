import random

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from general_util.logger import get_child_logger
from general_util.tokenization_utils import expand_special_tokenizer

logger = get_child_logger("Wiki.Path.seq2seq.collator")


def process_sent_dict(x):
    return " ".join(x["spans"])


def construct_seq2seq(example, generative_mode: bool = False):
    input_a = []
    input_b = []
    if "negative_context" in example:
        input_a.extend([example["condition"]] * (len(example["negative_context"]) + 1))
        input_b.extend([example["context"]] + example["negative_context"])
    else:
        op = [example["positive"]] + example["negative"]
        input_a.extend([example["context"]] * len(op))
        input_b.extend(op)

    input_a = [" ".join(map(process_sent_dict, x)) if isinstance(x, list) or isinstance(x, tuple) else process_sent_dict(x)
               for x in input_a]
    input_b = [" ".join(map(process_sent_dict, x)) if isinstance(x, list) or isinstance(x, tuple) else process_sent_dict(x)
               for x in input_b]
    assert len(input_a) == 4

    if generative_mode:
        return [input_a[0]], [input_b[0]]

    return input_a, input_b


class WikiSeq2SeqCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, causal_lm: bool = False, generative_mode: bool = False):
        self.max_seq_length = max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.causal_lm = causal_lm
        self.generative_mode = generative_mode

        expand_special_tokenizer(self.tokenizer)

    def __call__(self, batch):
        examples = [b["example"] for b in batch]
        inputs_a, inputs_b = [], []
        for exp in examples:
            res = construct_seq2seq(exp, generative_mode=self.generative_mode)
            assert len(res[0]) == len(res[1])
            if not self.causal_lm:
                inputs_a.extend(res[0])
                inputs_b.extend(res[1])
            else:
                inputs_a.extend(res[0])
                inputs_b.extend([x + " " + y + self.tokenizer.eos_token for x, y in zip(res[0], res[1])])

        batch_size = len(batch)
        op_num = len(inputs_a) // batch_size

        if not self.causal_lm:
            model_inputs = self.tokenizer(inputs_a, text_target=inputs_b, padding="longest", truncation=True, return_tensors="pt",
                                          max_length=self.max_seq_length)
            if not self.generative_mode:
                model_inputs["decoder_input_ids"] = model_inputs["labels"].reshape(batch_size, op_num, -1)
            # else:
            #     model_inputs["decoder_input_ids"] = model_inputs["labels"]
        else:
            tmp = self.tokenizer(inputs_a, padding="longest", truncation=True, return_tensors="pt", max_length=self.max_seq_length)
            # logger.info(tmp["input_ids"][0])
            # logger.info(self.tokenizer.decode(tmp["input_ids"][0], skip_special_tokens=False))
            input_lens = tmp["input_ids"].ne(self.tokenizer.pad_token_id).to(torch.long).sum(dim=-1)
            # logger.info(input_lens[0])
            model_inputs = self.tokenizer(inputs_b, padding="longest", truncation=True, return_tensors="pt", max_length=self.max_seq_length)
            # logger.info(model_inputs["input_ids"][0])
            # logger.info(self.tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=False))
            # logger.info("====================")
            # full_input_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).to(torch.long).sum(dim=-1)
            # for i in range(input_lens.size(0)):
            #     assert full_input_lens[i] > input_lens[i], (full_input_lens[i], input_lens[i])
            new_input_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1)
            input_lens = input_lens - input_lens.eq(new_input_lens).to(input_lens.dtype) * (input_lens // 2)
            input_lens = input_lens.to(torch.long)
            model_inputs["input_lens"] = input_lens

        if not self.generative_mode:
            model_inputs["input_ids"] = model_inputs["input_ids"].reshape(batch_size, op_num, -1)
            model_inputs["attention_mask"] = model_inputs["attention_mask"].reshape(batch_size, op_num, -1)
            model_inputs["labels"] = torch.zeros(len(examples), dtype=torch.long)

        return model_inputs


def flatten_options(options):
    return "\n".join(["{}: {}".format(chr(ord("A") + i), x) for i, x in enumerate(options)])


def construct_instruct_seq2seq(example, instruct: str, suffix: str):
    template = "{}\n\nContext:\n{}\n\nOptions:\n{}\n\n{}"

    if "negative_context" in example:
        context = example["condition"]
        options = [example["context"]] + example["negative_context"]
    else:
        context = example["context"]
        options = [example["positive"]] + example["negative"]

    context = " ".join(map(process_sent_dict, context)
                       ) if isinstance(context, list) or isinstance(context, tuple) else process_sent_dict(context)
    options = [" ".join(map(process_sent_dict, x)) if isinstance(x, list) or isinstance(x, tuple) else process_sent_dict(x)
               for x in options]
    options = list(enumerate(options))
    random.shuffle(options)
    label = -1
    for i, (idx, x) in enumerate(options):
        if idx == 0:
            label = i
            break
    assert label != -1
    # print(label)
    options = flatten_options([x[1] for x in options])
    input_a = template.format(instruct, context, options, suffix)
    input_b = chr(ord("A") + label)
    # print(input_b)

    return input_a, input_b


class WikiSeq2SeqInstructCollator:
    def __init__(self, max_seq_length: int, tokenizer: str, decoder_only: bool = False,
                 instruct: str = "Select the true option logically consistent with the context.",
                 suffix: str = "The answer is"):
        self.max_seq_length = max_seq_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        expand_special_tokenizer(self.tokenizer)
        self.instruct = instruct
        self.suffix = suffix
        self.decoder_only = decoder_only

    def __call__(self, batch):
        examples = [b["example"] for b in batch]
        inputs_a, inputs_b = [], []
        for exp in examples:
            res = construct_instruct_seq2seq(exp, self.instruct, self.suffix)
            inputs_a.append(res[0])
            if self.decoder_only:
                inputs_b.append(res[0] + " " + res[1])
            else:
                inputs_b.append(res[1])

        model_inputs = self.tokenizer(inputs_a, text_target=inputs_b, padding="longest", truncation=True, return_tensors="pt",
                                      max_length=self.max_seq_length)
        if self.decoder_only:
            input_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).to(torch.long).sum(dim=-1)
            model_inputs = self.tokenizer(inputs_b, padding="longest", truncation=True, return_tensors="pt",
                                          max_length=self.max_seq_length)
            model_inputs["input_lens"] = input_lens
            # tmp_input_lens = model_inputs["input_ids"].ne(self.tokenizer.pad_token_id).to(torch.long).sum(dim=-1)
            # assert (input_lens < tmp_input_lens).all(), (input_lens, tmp_input_lens)
        # print(self.tokenizer.decode(model_inputs["input_ids"][0]))
        return model_inputs

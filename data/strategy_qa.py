import json

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import TruncationStrategy, PaddingStrategy

from data.collators.dict2dict import DictTensorDataset
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def split_get_tensor_with_gold_para(file_path: str, tokenizer: PreTrainedTokenizer,
                                    train_para_file: str, max_seq_length: int, use_fact: bool = False):
    data = json.load(open(file_path))
    train_paragraphs = json.load(open(train_para_file))

    text_inputs_a = []
    text_inputs_b = []
    labels = []
    for item in data:
        question = item["question"]
        label = int(item["answer"])

        if use_fact:
            paragraphs = item["facts"]
        else:
            para_ids = set()
            for evidence in item["evidence"]:
                for annotation in evidence:
                    for evi_item in annotation:
                        if isinstance(evi_item, list):
                            for para_id in evi_item:
                                if para_id in train_paragraphs:
                                    # paragraphs.append(train_paragraphs[para_id]["content"])
                                    # Remove duplicate paragraphs.
                                    para_ids.add(para_id)
                                else:
                                    logger.warning(f"Cannot find paragraph with id: {para_id}")
                        else:
                            assert evi_item in ["operation", "no_evidence"], evi_item
            paragraphs = [train_paragraphs[para_id]["content"] for para_id in para_ids]

        context = " ".join(paragraphs)

        text_inputs_a.append(context)
        text_inputs_b.append(question)
        labels.append(label)

    model_inputs = tokenizer(text_inputs_a,
                             text_pair=text_inputs_b,
                             truncation=TruncationStrategy.LONGEST_FIRST,
                             padding=PaddingStrategy.LONGEST,
                             max_length=max_seq_length,
                             return_tensors="pt")
    model_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    dataset = DictTensorDataset(model_inputs)

    logger.info(f"Max seq length: {model_inputs['input_ids'].size(1)}")

    return dataset

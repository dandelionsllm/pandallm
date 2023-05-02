import json
import os
import re
from typing import Dict, List, Any, Union, Callable

import numpy as np
import torch
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin


class NumpySaver(DistGatherMixin):
    def __init__(self, save_copy: bool = False):
        self.predictions = []
        self.index = []
        self.save_copy = save_copy

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):

        logits = batch_model_outputs["logits"].detach().float()
        _, pred = logits.max(dim=-1)
        pred = pred.tolist()

        index = None
        if ddp:
            assert meta_data
            if isinstance(meta_data, list):
                index = [meta['index'].item() for meta in meta_data]
            elif isinstance(meta_data, dict):
                if isinstance(meta_data["index"], torch.Tensor):
                    index = meta_data["index"].tolist()
                else:
                    index = meta_data["index"]
            else:
                raise RuntimeError()
            obj = [pred, index]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                pred = []
                index = []
                for item in gather_res:
                    pred.extend(item[0])
                    index.extend(item[1])

        if index is not None:
            self.index.extend(index)
        self.predictions.extend(pred)

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.npy")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.npy")

        if len(self.index):
            assert len(self.index) == len(self.predictions)
            predictions = {idx: pred for idx, pred in zip(self.index, self.predictions)}
            predictions = sorted(predictions.items(), key=lambda x: x[0])
            predictions = [pred[1] for pred in predictions]
            np.save(output_file, np.array(predictions))
        else:
            np.save(output_file, np.array(self.predictions))

        if self.save_copy:
            if dist.is_initialized():
                output_file = os.path.join(output_dir, f"eval_predictions_copy_rank{dist.get_rank()}.bin")
            else:
                output_file = os.path.join(output_dir, "eval_predictions_copy.bin")

            torch.save({
                "index": self.index,
                "predictions": self.predictions
            }, output_file)

        return {}, self.predictions


class TaggingSaver(DistGatherMixin):
    def __init__(self):
        self.logits = []
        self.index = []

    def __call__(self, meta_data: List[Dict[str, Any]], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        tagging_logits = batch_model_outputs["tagging_logits"].detach().float().tolist()

        index = None
        if ddp:
            assert meta_data
            if isinstance(meta_data, list):
                index = [meta['index'].item() for meta in meta_data]
            elif isinstance(meta_data, dict):
                index = meta_data["index"].tolist()
            else:
                raise RuntimeError()
            obj = [tagging_logits, index]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tagging_logits = []
                index = []
                for item in gather_res:
                    tagging_logits.extend(tagging_logits)
                    index.extend(item[1])

        if index is not None:
            self.index.extend(index)
        self.logits.extend(tagging_logits)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"tagging_logits_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "tagging_logits.json")

        if len(self.index):
            assert len(self.index) == len(self.logits)
            predictions = {idx: pred for idx, pred in zip(self.index, self.logits)}
            predictions = sorted(predictions.items(), key=lambda x: x[0])
            predictions = [pred[1] for pred in predictions]
            json.dump(predictions, open(output_file, "w"))
        else:
            json.dump(self.logits, open(output_file, "w"))

        return {}, self.logits


def answer_clean(pred_seq: str, reverse: bool = False, answer_trigger: str = "The answer is"):
    if answer_trigger:
        pred_seq = pred_seq.split(answer_trigger)[1]
    # pred = re.findall(r'A|B|C|D|E', pred_seq)
    pred = re.findall(r'A|B|C|D', pred_seq)
    if len(pred) == 0:
        return ""
    if reverse:
        return pred[-1]
    return pred[0]


class GeneratorPredictor(DistGatherMixin):
    def __init__(self, reverse: bool = False, answer_trigger: str = "The answer is"):
        self.predictions = []
        self.reverse = reverse
        self.answer_trigger = answer_trigger

    def __call__(self, meta_data: Union[List[Dict[str, Any]], Dict[str, Any]], batch_model_outputs, ddp: bool = False):
        labels = meta_data["label"]
        prompt_index = meta_data["prompt_index"]
        index = meta_data["index"].tolist()
        inputs = meta_data["input"]

        pred_seq = batch_model_outputs["generated_seq"]
        assert len(labels) == len(prompt_index) == len(index) == len(inputs), (len(labels), len(prompt_index), len(index), len(inputs))
        if len(pred_seq) == len(labels):
            pass
        elif len(pred_seq) % len(labels) == 0:
            pass
        else:
            raise ValueError((len(pred_seq), len(labels)))

        predictions = [
            {
                "label": label,
                "index": idx,
                "prompt_index": prompt_idx,
                "output": res,
                "cleaned_output": answer_clean(res, self.reverse, self.answer_trigger),
                "input": src,
            } for label, idx, prompt_idx, res, src in zip(labels, index, prompt_index, pred_seq, inputs)
        ]

        if ddp:
            gather_res = self.gather_object(predictions)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"decode_results_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "decode_results.json")

        json.dump(self.predictions, open(output_file, "w"))
        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["label"] == pred["cleaned_output"]:
                correct += 1
            if pred["cleaned_output"]:
                npy_outputs.append(ord(pred["cleaned_output"]) - ord("A"))
            else:
                npy_outputs.append(0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return {"acc": correct / len(existing_ids)}, self.predictions


class GeneratorPredictorV2(DistGatherMixin):
    def __init__(self, answer_cleaner: Callable):
        self.predictions = []
        self.answer_cleaner = answer_cleaner

    def __call__(self, meta_data: Union[List[Dict[str, Any]], Dict[str, Any]], batch_model_outputs, ddp: bool = False):
        labels = meta_data["label"]
        prompt_index = meta_data["prompt_index"]
        index = meta_data["index"].tolist()
        inputs = meta_data["input"]

        pred_seq = batch_model_outputs["generated_seq"]
        assert len(labels) == len(prompt_index) == len(index) == len(inputs), (len(labels), len(prompt_index), len(index), len(inputs))
        if len(pred_seq) == len(labels):
            pass
        elif len(pred_seq) % len(labels) == 0:
            pass
        else:
            raise ValueError((len(pred_seq), len(labels)))

        predictions = [
            {
                "label": label,
                "index": idx,
                "prompt_index": prompt_idx,
                "output": res,
                "cleaned_output": self.answer_cleaner(res, src),
                "input": src,
            } for label, idx, prompt_idx, res, src in zip(labels, index, prompt_index, pred_seq, inputs)
        ]

        if ddp:
            gather_res = self.gather_object(predictions)
            if dist.get_rank() == 0:
                tmp = []
                for item in gather_res:
                    tmp.extend(item)
                predictions = tmp

        self.predictions.extend(predictions)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"decode_results_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "decode_results.json")

        json.dump(self.predictions, open(output_file, "w"))
        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        correct = 0
        existing_ids = set()
        npy_outputs = []
        for pred in self.predictions:
            if pred["index"] in existing_ids:
                continue
            existing_ids.add(pred["index"])
            if pred["label"] == pred["cleaned_output"]:
                correct += 1
            if pred["cleaned_output"] and len(pred["cleaned_output"]) == 1:
                npy_outputs.append(ord(pred["cleaned_output"]) - ord("A"))
            else:
                npy_outputs.append(0)

        metrics = {"acc": correct / len(existing_ids)}

        if not dist.is_initialized() or dist.get_rank() == 0:
            np.save(os.path.join(output_dir, "decode_results.npy"), np.array(npy_outputs))
            json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"))
        assert len(npy_outputs) == len(existing_ids), (len(npy_outputs), len(self.predictions), len(existing_ids))
        return metrics, self.predictions

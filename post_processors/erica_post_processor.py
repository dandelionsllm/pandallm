import os
from typing import Dict, List, Any, Tuple

import torch
import torch.distributed as dist

from general_util.logger import get_child_logger
from post_processors.dist_mixin import DistGatherMixin

logger = get_child_logger(__name__)


class ERICAPredictionSaver(DistGatherMixin):
    def __init__(self):
        self.indices = []
        self.entity_mentions = []
        self.entity_hidden = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        entity_mentions: List[List[Dict]] = meta_data["entity_mentions"]
        entity_spans: List[Dict[str, List[Tuple[int, int]]]] = meta_data["entity_spans"]
        indices = meta_data["index"]
        ent_hidden = batch_model_outputs["entity_hidden"]

        flat_entity_rep = []
        for b, b_entity_spans in enumerate(entity_spans):
            item_entity_reps = {}
            for j, ent_id in enumerate(b_entity_spans.keys()):
                assert ent_id not in item_entity_reps
                item_entity_reps[ent_id] = ent_hidden[b, j]
            flat_entity_rep.append(item_entity_reps)

        # if ddp:
        #     obj = [flat_entity_rep, entity_mentions, indices]
        #     gather_res = self.gather_object(obj)
        #     if dist.get_rank() == 0:
        #         tmp_a = []
        #         tmp_b = []
        #         tmp_c = []
        #         for item in gather_res:
        #             tmp_a.extend(item[0])
        #             tmp_b.extend(item[1])
        #             tmp_c.extend(item[2])
        #         flat_entity_rep = tmp_a
        #         entity_mentions = tmp_b
        #         indices = tmp_c

        self.entity_hidden.extend(flat_entity_rep)
        self.entity_mentions.extend(entity_mentions)
        self.indices.extend(indices)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"predictions-rank{dist.get_rank()}.pt")
        else:
            output_file = os.path.join(output_dir, f"predictions.pt")
        torch.save({
            "mentions": self.entity_mentions,
            "hidden": self.entity_hidden,
            "index": self.indices,
        },
            output_file)
        return {}, []


class WikiPathInferencePostProcessor(DistGatherMixin):
    def __init__(self):
        self.indices = []
        self.codes = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        indices = meta_data["indices"]
        codes = batch_model_outputs["code_indices"].tolist()

        if ddp:
            obj = [codes, indices]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                tmp_a = []
                tmp_b = []
                for item in gather_res:
                    tmp_a.extend(item[0])
                    tmp_b.extend(item[1])
                codes = tmp_a
                indices = tmp_b

        self.indices.extend(indices)
        self.codes.extend(codes)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"predictions-rank{dist.get_rank()}.pt")
        else:
            output_file = os.path.join(output_dir, f"predictions.pt")
        torch.save({
            "indices": self.indices,
            "codes": self.codes,
        }, output_file)
        logger.info(f"Index num: {len(self.indices)}\tCode num:{len(self.codes)}")
        return {}, []


class CausalLMInferencePostProcessor(DistGatherMixin):
    def __init__(self):
        self.indices = []
        self.losses = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        indices = meta_data["index"]
        losses = batch_model_outputs["loss"].tolist()

        # if ddp:
        #     obj = [losses, indices]
        #     gather_res = self.gather_object(obj)
        #     if dist.get_rank() == 0:
        #         tmp_a = []
        #         tmp_b = []
        #         for item in gather_res:
        #             tmp_a.extend(item[0])
        #             tmp_b.extend(item[1])
        #         losses = tmp_a
        #         indices = tmp_b

        self.indices.extend(indices)
        self.losses.extend(losses)

    def get_results(self, output_dir: str):
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"predictions-rank{dist.get_rank()}.pt")
        else:
            output_file = os.path.join(output_dir, f"predictions.pt")
        torch.save({
            "indices": self.indices,
            "losses": self.losses,
        }, output_file)
        logger.info(f"Index num: {len(self.indices)}\tLoss num:{len(self.losses)}")
        return {}, []

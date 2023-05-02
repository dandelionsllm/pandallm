"""
Write your own your own collators under the directory.
"""

from typing import Dict, Union, Any, List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers.tokenization_utils import BatchEncoding


class DictTensorDataset(Dataset):
    def __init__(self, data: Union[Dict[str, Tensor], BatchEncoding], meta_data: List[Dict[str, Any]] = None):
        self.data = data
        self.meta_data = meta_data
        self.keys = list(self.data.keys())
        for v in self.data.values():
            if meta_data is not None:
                assert len(v) == len(meta_data)
            else:
                assert len(v) == self.data[self.keys[0]].size(0)

    def __len__(self):
        return self.data[self.keys[0]].size(0)

    def __getitem__(self, idx):
        res = {k: v[idx] for k, v in self.data.items()}
        if self.meta_data is not None:
            res["meta_data"] = self.meta_data[idx]
        if "index" not in res or "index" not in res["meta_data"]:
            res["index"] = torch.LongTensor([idx])
        return res


class MetaCollator:
    def __call__(self, batch):
        if "meta_data" not in batch[0]:
            return default_collate(batch)

        meta_data = [b.pop("meta_data") for b in batch]
        batch = default_collate(batch)
        batch["meta_data"] = meta_data
        return batch

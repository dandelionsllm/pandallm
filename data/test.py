import os
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, file_path, tokenizer, pseudo_dataset_len: int = -1):
        super().__init__()
        self.data = ["My name is Jiao Fangkai."]
        self.pseudo_dataset_len = pseudo_dataset_len
        # print("============================", os.environ["LOCAL_RANK"], "Test dataset initialized.")

    def __len__(self):
        if self.pseudo_dataset_len > 0:
            return self.pseudo_dataset_len
        return 100000000

    def __getitem__(self, index):
        return {
            "flan": {
                "inputs": self.data[0],
                "targets": self.data[0],
            },
            "index": index,
        }

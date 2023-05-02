import json
import random
from glob import glob
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger("ZN_INSTRUCT")


def filter_train(file_path):
    files = list(glob(file_path, recursive=True))
    new_files = []
    for file in files:
        if "valid" in file or "test" in file:
            continue
        new_files.append(file)
    return sorted(new_files)


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        logger.info(f"Loading data from {file_path}")
        self.data = []
        if 'coig_data.json' in file_path:
            data = json.load(open(file_path, 'r'))
            for item in data:
                if isinstance(item, list):
                    history = ""
                    for turn_id, turn in enumerate(item):
                        self.data.append({
                            "inputs": history + "\n\n" + turn["inputs"],
                            "targets": turn["targets"],
                        })
                        if turn_id == 0:
                            history += turn["inputs"] + "\n\n" + turn["targets"]
                        else:
                            history += "\n\n" + turn["inputs"] + "\n\n" + turn["targets"]
                else:
                    self.data.append(item)

        else:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line[-1] == '\n':
                        line = line[:-1]

                    item = json.loads(line)
                    if "text" in item and item["text"].strip():
                        self.data.append({
                            "inputs": " ",
                            "targets": item["text"],
                        })
                    else:
                        inputs = ""
                        if "instruction" in item:
                            if isinstance(item["instruction"], list):
                                inputs += random.choice(item["instruction"])
                            else:
                                inputs += item["instruction"]
                        if "input" in item:
                            inputs += " " + item["input"]
                        if not item["output"].strip():
                            continue
                        self.data.append({
                            "inputs": inputs,
                            "targets": item["output"],
                        })

        logger.info(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "flan": self.data[idx],
        }


class TextDatasetUnify(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer):
        glob_path = file_path + "/**/*.json"
        files = filter_train(glob_path)
        self.data = []
        for file in files:
            cur_num = len(self.data)
            logger.info(f"Loading data from {file}")
            if 'coig_data.json' in file:
                data = json.load(open(file, 'r'))
                for item in data:
                    if isinstance(item, list):
                        history = ""
                        for turn_id, turn in enumerate(item):
                            self.data.append({
                                "inputs": history + "\n\n" + turn["inputs"],
                                "targets": turn["targets"],
                            })
                            if turn_id == 0:
                                history += turn["inputs"] + "\n\n" + turn["targets"]
                            else:
                                history += "\n\n" + turn["inputs"] + "\n\n" + turn["targets"]
                    else:
                        self.data.append(item)
            else:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line[-1] == '\n':
                            line = line[:-1]

                        item = json.loads(line)
                        if "text" in item and item["text"].strip():
                            self.data.append({
                                "inputs": " ",
                                "targets": item["text"],
                            })
                        else:
                            inputs = ""
                            if "instruction" in item:
                                if isinstance(item["instruction"], list):
                                    inputs += random.choice(item["instruction"])
                                else:
                                    inputs += item["instruction"]
                            if "input" in item:
                                inputs += " " + item["input"]
                            if not item["output"].strip():
                                continue
                            self.data.append({
                                "inputs": inputs,
                                "targets": item["output"],
                            })

            logger.info(f"Loaded {len(self.data) - cur_num} examples.")

        logger.info(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "flan": self.data[idx],
        }

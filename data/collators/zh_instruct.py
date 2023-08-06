import json
import random
from glob import glob
from tqdm import tqdm
import gzip

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List

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


def unified_conversion(file):
    all_data = []
    if 'coig_data.json' in file:
        data = json.load(open(file, 'r'))
        for item in data:
            if isinstance(item, list):
                history = ""
                for turn_id, turn in enumerate(item):
                    all_data.append({
                        "inputs": history + "\n\n" + turn["inputs"],
                        "targets": turn["targets"],
                    })
                    if turn_id == 0:
                        history += turn["inputs"] + "\n\n" + turn["targets"]
                    else:
                        history += "\n\n" + turn["inputs"] + "\n\n" + turn["targets"]
            else:
                all_data.append(item)
    elif "WuDaoCorpus" in file:
        data = json.load(open(file, "r"))
        for item in data:
            all_data.append({
                "inputs": "",
                "targets": item["title"] + " " + item["content"],
            })
    elif "c4" in file:
        with gzip.open(file, "rb") as f:
            for line in f:
                item = json.loads(line)
                all_data.append({
                    "inputs": "",
                    "targets": item["text"],
                })
    else:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[-1] == '\n':
                    line = line[:-1]

                item = json.loads(line)
                if "text" in item and item["text"].strip():
                    all_data.append({
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
                    all_data.append({
                        "inputs": inputs,
                        "targets": item["output"],
                    })
    return all_data


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
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 exclude_path: List[str] = None, wudao_sample_list: str = None):
        glob_path = file_path + "/**/*.json"
        files = filter_train(glob_path)

        if wudao_sample_list is not None:
            wudao_include_files = set(json.load(open(wudao_sample_list, "r")))
        else:
            wudao_include_files = {}

        self.data = []
        for file in files:
            if exclude_path is not None and any([path in file for path in exclude_path]):
                continue
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
            elif "WuDaoCorpus" in file:
                if file not in wudao_include_files:
                    continue
                data = json.load(open(file, "r"))
                for item in data:
                    self.data.append({
                        "inputs": "",
                        "targets": item["title"] + " " + item["content"],
                    })
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


def compose_input_output(item):
    _input = ""
    if "instruction" in item:
        if isinstance(item["instruction"], list):
            _input += random.choice(item["instruction"])
        else:
            _input += item["instruction"]
    if "input" in item:
        if _input:
            _input += " " + item["input"]
        else:
            _input += item["input"]
    if not item["target"].strip():
        return None
    return {
        "inputs": _input,
        "targets": item["target"],
    }


class TextDatasetCombineV2(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, extra_data: Dataset = None):
        glob_path = file_path + "/**/*.json"
        files = filter_train(glob_path)
        self.tokenizer = tokenizer
        self.data = []
        for file in files:
            data = json.load(open(file, 'r'))
            logger.info(f"Loading data from {file}")
            for item in data:
                if isinstance(item, dict):
                    new_item = compose_input_output(item)
                    if new_item is not None:
                        self.data.append(new_item)
                # elif isinstance(item, list):
                #     history = ""
                #     for turn_id, turn in enumerate(item):
                #         new_item = compose_input_output(turn)
                #         if new_item is not None:
                #             self.data.append({
                #                 "inputs": history + "\n" + new_item["inputs"],
                #                 "targets": new_item["targets"],
                #             })
                #             if turn_id == 0:
                #                 history += new_item["inputs"] + "\n" + new_item["targets"]
                #             else:
                #                 history += "\n" + new_item["inputs"] + "\n" + new_item["targets"]
                # Do not every single turn because there are too many turns
                elif isinstance(item, list):
                    new_item = compose_input_output(item[0])
                    if new_item is None:
                        continue
                    _input = new_item["inputs"]
                    _target = new_item["targets"] + self.tokenizer.eos_token

                    for turn in item[1:]:
                        new_item = compose_input_output(turn)
                        if new_item is None:
                            continue
                        _target += new_item["inputs"] + "\n" + new_item["targets"] + self.tokenizer.eos_token

                    _target = _target[:-len(self.tokenizer.eos_token)]
                    self.data.append({
                        "inputs": _input,
                        "targets": _target,
                    })
                else:
                    raise ValueError(f"Unknown data type: {type(item)}")

        logger.info(f"Loaded {len(self.data)} examples.")
        self.extra_data = extra_data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def filter_keywords(text):
        text = text.replace("openai", "PandaLLM").replace("OpenAI", "PandaLLM")
        text = text.replace("Fudan University", "Nanyang Technological University").replace("FDU", "NTU")
        text = text.replace("MOSS", "PandaLLM")
        return text

    def __getitem__(self, idx):
        res = {
            "flan": {
                "inputs": self.filter_keywords(self.data[idx]["inputs"]),
                "targets": self.filter_keywords(self.data[idx]["targets"]),
            },
        }
        if self.extra_data is not None:
            res["extra"] = self.extra_data[idx % len(self.extra_data)]["flan"]
        return res


class WuDaoCorpusDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, file_num: int = -1):
        glob_path = file_path + "/*.json"
        files = sorted(list(glob(glob_path)))
        if file_num > 0:
            files = files[:file_num]
        self.data = []
        for file in tqdm(files, desc=f"Loading WuDao Corpus from {file_path}"):
            data = json.load(open(file, "r"))
            for item in data:
                self.data.append({
                    "inputs": "",
                    "targets": item["title"] + " " + item["content"],
                })

        logger.info(f"Loaded {len(self.data)} Wudao examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "flan": self.data[idx],
        }


class C4CorpusDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, file_num: int = -1):
        super().__init__()
        glob_path = file_path + "/*.json.gz"
        files = sorted(list(glob(glob_path)))
        if file_num > 0:
            files = files[:file_num]
        self.data = []
        for file in tqdm(files, desc=f"Loading C4 Corpus from {file_path}"):
            with gzip.open(file, "rb") as f:
                for line in f:
                    item = json.loads(line)
                    self.data.append({
                        "inputs": "",
                        "targets": item["text"],
                    })

        logger.info(f"Loaded {len(self.data)} C4 examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "flan": self.data[idx],
        }


class TextDatasetUnifyV3(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, pair_file_list: str):
        self.tokenizer = tokenizer
        self.data = []
        file_list = json.load(open(file_path, "r"))
        for file in file_list:
            logger.info(f"Loading data from {file}")
            self.data.extend(unified_conversion(file))

        if pair_file_list is not None:
            self.pair_data = []
            pair_file_list = sorted(list(glob(pair_file_list, recursive=True)))
            pair_file = random.choice(pair_file_list)
            pair_file = json.load(open(pair_file, "r"))
            for file in pair_file:
                logger.info(f"Loading data from {file}")
                self.pair_data.extend(unified_conversion(file))
        else:
            self.pair_data = None

        logger.info(f"Loaded {len(self.data)} examples.")
        if self.pair_data is not None:
            logger.info(f"Loaded {len(self.pair_data)} pair examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = {
            "flan": self.data[idx],
        }
        if self.pair_data is not None:
            res["extra"] = self.pair_data[idx % len(self.pair_data)]
        return res

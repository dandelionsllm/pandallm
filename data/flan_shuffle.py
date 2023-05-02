import argparse
import collections
import glob
from tqdm import trange
import json
import os.path
import random

import torch

cot_zs_submix = (
    ("cot_zs_noopt_train.jsonl.gz", 1),
    ("cot_zs_opt_train.jsonl.gz", 1),
)

dialog_zs_submix = (
    ("dialog_zs_noopt_train.jsonl.gz", 1),
    ("dialog_zs_opt_train.jsonl.gz", 1),
)

flan_zs_submix = (
    ("flan_zs_noopt_train.jsonl.gz", 1),
    ("flan_zs_opt_train.jsonl.gz", 1),
)

niv2_zs_submix = (
    ("niv2_zs_noopt_train.jsonl.gz", 1),
    ("niv2_zs_opt_train.jsonl.gz", 1),
)

t0_zs_submix = (
    ("t0_zs_noopt_train.jsonl.gz", 1),
    ("t0_zs_opt_train.jsonl.gz", 1),
)

# flan_v2_submix = (
#     (flan_zs_submix, 0.4),  # mixing weight = 40%
#     (t0_zs_submix, 0.32),  # mixing weight = 32%
#     (niv2_zs_submix, 0.2),  # mixing weight = 20%
#     (cot_zs_submix, 0.05),  # mixing weight = 5%
#     (dialog_zs_submix, 0.03),  # mixing weight = 3%
# )

flan_v2_submix = {
    "flan": 0.4,
    "t0": 0.32,
    "niv2": 0.2,
    "cot": 0.05,
    "dialog": 0.03,
}
flan_v2_datasets = {
    "flan": [
        "flan_zs_noopt_train.jsonl",
        "flan_zs_opt_train.jsonl",
        "flan_fs_noopt_train.jsonl",
        "flan_fs_opt_train.jsonl",
    ],
    "t0": [
        "t0_zs_noopt_train.jsonl",
        "t0_zs_opt_train.jsonl",
        "t0_fs_noopt_train.jsonl",
        # "t0_fs_opt_train.jsonl",
    ],
    "niv2": [
        # "niv2_zs_noopt_train.jsonl",
        "niv2_zs_opt_train.jsonl",
        # "niv2_fs_noopt_train.jsonl",
        "niv2_fs_opt_train.jsonl",
    ],
    "cot": [
        # "cot_zs_noopt_train.jsonl",
        "cot_zs_opt_train.jsonl",
        # "cot_fs_noopt_train.jsonl",
        "cot_fs_opt_train.jsonl",
    ],
    "dialog": [
        # "dialog_zs_noopt_train.jsonl",
        "dialog_zs_opt_train.jsonl",
        # "dialog_fs_noopt_train.jsonl",
        "dialog_fs_opt_train.jsonl",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="data")
parser.add_argument("--output_dir", type=str, default="data")
parser.add_argument("--cot_resampling", default=False, action="store_true")
parser.add_argument("--niv2_resampling", default=False, action="store_true")
parser.add_argument("--split_size", type=int, default=10000000)
args = parser.parse_args()

file_readers = collections.defaultdict(dict)
for prefix, datasets in flan_v2_datasets.items():
    for dataset in datasets:
        file = os.path.join(args.input_dir, dataset)
        file_readers[prefix][file] = open(file, "r")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

split_id = 0
while True:
    data_slit = []
    for prefix, ratio in flan_v2_submix.items():
        data_num = int(args.split_size * ratio)
        dataset_num = len(file_readers[prefix])
        if dataset_num == 0:
            continue
        sub_dataset_num = data_num // dataset_num
        remove_flag = {file: False for file in file_readers[prefix].keys()}
        for file, reader in file_readers[prefix].items():
            print(f"Reading {file} ...")
            for _ in trange(sub_dataset_num):
                line = reader.readline()
                if line:
                    data_slit.append(json.loads(line))
                else:
                    remove_flag[file] = True
                    break
        for file, flag in remove_flag.items():
            if flag:
                tmp = file_readers[prefix].pop(file)
                tmp.close()
            if args.cot_resampling and prefix == "cot":
                file_readers[prefix][file] = open(file, "r")
            if args.niv2_resampling and prefix == "niv2":
                file_readers[prefix][file] = open(file, "r")

    if len(data_slit) == 0 or len(file_readers["flan"]) == 0:
        break
    random.shuffle(data_slit)
    torch.save(data_slit, os.path.join(args.output_dir, f"flan_v2_{split_id}.pt"))
    split_id += 1
    del data_slit

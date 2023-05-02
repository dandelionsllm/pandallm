import argparse
import gzip
import json
import os.path

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

flan_v2_submix = (
    (flan_zs_submix, 0.4),  # mixing weight = 40%
    (t0_zs_submix, 0.32),  # mixing weight = 32%
    (niv2_zs_submix, 0.2),  # mixing weight = 20%
    (cot_zs_submix, 0.05),  # mixing weight = 5%
    (dialog_zs_submix, 0.03),  # mixing weight = 3%
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--total_data_num", type=int, default=500000)
    parser.add_argument("--output_file", type=str, default="data")
    args = parser.parse_args()

    all_data = []
    for dataset, ratio in flan_v2_submix:
        data_num = int(args.total_data_num * ratio)
        dataset_data = []
        tmp = sum([sub_ratio for _, sub_ratio in dataset])

        for file, sub_ratio in dataset:
            sub_data_num = int(data_num * sub_ratio / tmp)

            sub_data_f = gzip.open(os.path.join(args.input_dir, file), "rt")
            sub_data = []
            line = sub_data_f.readline()
            while line:
                sub_data.append(json.loads(line))
                if len(sub_data) >= sub_data_num:
                    break
                line = sub_data_f.readline()
            sub_data_f.close()
            print(f"Read {len(sub_data)} lines from {file}")

            sub_data = sub_data[:sub_data_num]
            print(sub_data[0])
            print(sub_data[-1])
            dataset_data.extend(sub_data)

        all_data.extend(dataset_data)

    print(f"Total data num: {len(all_data)}")
    torch.save(all_data, args.output_file)

data_group = [
    [
        "cot_fs_noopt_train.jsonl.gz",
        "cot_fs_opt_train.jsonl.gz",
        "cot_zs_noopt_train.jsonl.gz",
        "cot_zs_opt_train.jsonl.gz",
        "niv2_fs_noopt_train.jsonl.gz",
        "niv2_fs_opt_train.jsonl.gz",
        "niv2_zs_noopt_train.jsonl.gz",
        "niv2_zs_opt_train.jsonl.gz",
    ],
    [
        "dialog_zs_noopt_train.jsonl.gz",
        "dialog_zs_opt_train.jsonl.gz",
    ],
    "dialog_fs_noopt_train.jsonl.gz",
    "dialog_fs_opt_train.jsonl.gz",
    "flan_fs_noopt_train.jsonl.gz",
    "flan_fs_opt_train_part1.jsonl.gz",
    "flan_fs_opt_train_part2.jsonl.gz",
    "flan_fs_opt_train_part3.jsonl.gz",
    "flan_zs_noopt_train.jsonl.gz",
    "flan_zs_opt_train.jsonl.gz",
    "t0_fs_noopt_train.jsonl.gz",
    "t0_zs_noopt_train.jsonl.gz",
    "t0_zs_opt_train.jsonl.gz",
]


def obtain_flan_collection_group():
    return data_group

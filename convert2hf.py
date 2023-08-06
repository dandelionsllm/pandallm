import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

PARAM_MAP = {
    "7B": {
        "n_layers": 32,
    },
    "13B": {
        "n_layers": 40,
    },
    "30B": {
        "n_layers": 60,
    },
    "65B": {
        "n_layers": 80,
    },
}

ORIGINAL_TOKENIZER_SIZE = 32000


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def load_weights(checkpoint_dir, n_layers: int):
    state_dict = {}
    for pt in Path(checkpoint_dir).iterdir():
        print("Processing ", pt.name)
        if not pt.name.startswith('layer_'):
            continue

        sd = torch.load(pt, map_location="cpu")

        if pt.name.startswith("layer_00"):
            print(f"{pt.name} -> model.embed_tokens.weight")
            state_dict["model.embed_tokens.weight"] = sd["weight"]
        elif pt.name.startswith(f"layer_{n_layers + 1}"):
            print(f"{pt.name} -> model.norm.weight")
            state_dict["model.norm.weight"] = sd["weight"]
        elif pt.name.startswith(f"layer_{n_layers + 2}"):
            print(f"{pt.name} -> lm_head.weight")
            state_dict["lm_head.weight"] = sd["weight"]
        else:
            layer_idx = int(pt.name[len("layer_"):].split("-")[0]) - 1
            assert 0 <= layer_idx < n_layers
            for k, v in sd.items():
                state_dict[f"model.layers.{layer_idx}.{k}"] = v
            print(f"{pt.name} -> model.layers.{layer_idx}")
    return state_dict


def write_model(input_base_path, model_size, config_dir):
    assert model_size in PARAM_MAP
    config = transformers.AutoConfig.from_pretrained(config_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    params = PARAM_MAP[model_size]
    n_layers = params["n_layers"]

    if os.path.exists(input_base_path):
        checkpoint_dirs = [input_base_path]
    else:
        checkpoint_dirs = glob(input_base_path, recursive=True)
    print(f"Found checkpoints: {checkpoint_dirs}")

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_state_dict = load_weights(checkpoint_dir, n_layers)
        model.save_pretrained("/".join(checkpoint_dir.split("/")[:-1]), state_dict=checkpoint_state_dict, max_shard_size="3GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B"],
    )
    parser.add_argument(
        "--config_dir",
    )
    args = parser.parse_args()
    write_model(
        input_base_path=args.input_dir,
        model_size=args.model_size,
        config_dir=args.config_dir,
    )


if __name__ == "__main__":
    main()

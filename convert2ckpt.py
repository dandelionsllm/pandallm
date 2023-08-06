from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from general_util.tokenization_utils import expand_special_tokenizer, PreTrainedTokenizer


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="/path/to/llama-7b-hf")
    output_dir: str = field(default="./llama-7B-init-ckpt")
    mp_world_size: int = field(default=1)


def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # TODO: padding embedding size for being divisible by 64.
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    num_new_tokens = len(tokenizer) - original_vocab_size
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def write_ckpt(outpath: Path, model: torch.nn.Module, model_config: LlamaConfig, mp: int):
    loaded = model.state_dict()

    n_layers = model_config.num_hidden_layers
    # embedding
    sd = {"weight": loaded['model.embed_tokens.weight']}
    torch.save(sd, outpath / "layer_00-model_00-model_states.pt")
    # norm
    sd = {f"weight": loaded['model.norm.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 1}-model_00-model_states.pt")
    # lm head
    sd = {f"weight": loaded['lm_head.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 2}-model_00-model_states.pt")
    # decoder layers
    for layer_i in range(n_layers):
        sd = {nm.replace(f"model.layers.{layer_i}.", f""): weight for nm, weight in loaded.items() if
              nm.startswith(f"model.layers.{layer_i}.")}
        torch.save(sd, outpath / f"layer_{layer_i + 1:02d}-model_00-model_states.pt")

    model_state = {
        "dp_world_size": 1,
        "mp_world_size": mp,
        "module": None,
        "optimizer": None,
        "global_steps": 1,
        "skipped_steps": 1,
        "iteration": 1,
    }
    for rank in range(mp):
        torch.save(model_state, outpath / f"mp_rank_{rank:02d}_model_states.pt")


def main():
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    tokenizer: PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model_config = model.config

    original_vocab_size = model_config.vocab_size
    tokenizer.add_tokens(["<eot>", "<ext_0>", "<ext_1>", "<ext_2>", "<ext_3>"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if len(tokenizer) > original_vocab_size:
        print(f"expand vocab size from {original_vocab_size} to {len(tokenizer)}")
        smart_tokenizer_and_embedding_resize(tokenizer, model)

    outpath = Path(args.output_dir)
    if outpath.exists():
        print(f"{outpath} exists. Do nothing.")
        exit(0)

    print(f"create {outpath}")
    outpath.mkdir()
    steppath = outpath / "global_step001"
    steppath.mkdir()

    write_ckpt(steppath, model, model_config, args.mp_world_size)
    with open(outpath / "latest", "w") as fout:
        fout.write("global_step001")

    tokenizer.save_pretrained(outpath)
    model_config.save_pretrained(outpath)


if __name__ == "__main__":
    main()

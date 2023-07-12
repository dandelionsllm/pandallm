from typing import Union, Optional, Dict

import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import BatchEncoding

from models.mpt.modeling_mpt import attn_bias_shape, build_attn_bias, MPTConfig


def get_lm_labels(input_lens, input_ids, pad_token_id, ignore_index=-100):
    labels = input_ids.clone()

    label_mask = labels.ne(pad_token_id)
    lens_mask = torch.arange(labels.size(1))[None, :] >= input_lens[:, None]
    label_mask = label_mask & lens_mask

    labels = labels.masked_fill(~label_mask, ignore_index).contiguous()

    return labels


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            torch.float16,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, torch.float16, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def convert_to_standard_inputs(model_inputs: Dict, tokenizer: PreTrainedTokenizer, ignored_index: int = -100):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    # input_lens = getattr(model_inputs, "input_lens", None)
    input_lens = model_inputs["input_lens"]

    labels = get_lm_labels(input_lens, input_ids, tokenizer.pad_token_id, ignored_index)

    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    attention_mask = _prepare_decoder_attention_mask(attention_mask, input_ids.shape, 0)

    return input_ids, attention_mask, position_ids, labels


def _attn_bias(config: MPTConfig, dtype=torch.float32, ):
    attn_impl = config.attn_config["attn_impl"]
    alibi = config.attn_config['alibi']
    prefix_lm = config.attn_config['prefix_lm']
    is_causal = not prefix_lm
    attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
    alibi_bias_max = config.attn_config['alibi_bias_max']

    _attn_bias_shape = attn_bias_shape(attn_impl, config.n_heads, config.max_seq_len, alibi, prefix_lm=prefix_lm,
                                       causal=is_causal, use_sequence_id=attn_uses_sequence_id)
    attn_bias = None
    if _attn_bias_shape:
        attn_bias = torch.zeros(_attn_bias_shape, dtype=dtype)
        attn_bias = build_attn_bias(attn_impl, attn_bias, config.n_heads, config.max_seq_len,
                                    causal=is_causal, alibi=alibi, alibi_bias_max=alibi_bias_max)

    return attn_bias


class LlamaPpInputsProcess:
    def __call__(self, model_inputs: Union[Dict, BatchEncoding], tokenizer: PreTrainedTokenizer):
        input_ids, attention_mask, position_ids, labels = convert_to_standard_inputs(model_inputs, tokenizer)
        return (
            (input_ids, attention_mask, position_ids),
            labels
        )


class MPTPpInputsProcess:
    def __init__(self, config: MPTConfig, dtype=torch.float32):
        self.config = config
        self.dtype = dtype
        self.attn_bias = _attn_bias(config, dtype)

    def __call__(self, model_inputs: Union[Dict, BatchEncoding], tokenizer: PreTrainedTokenizer):
        input_ids, _, _, labels = convert_to_standard_inputs(model_inputs, tokenizer)
        attention_mask = model_inputs["attention_mask"]

        return (
            (input_ids, attention_mask, self.attn_bias),
            labels,
        )

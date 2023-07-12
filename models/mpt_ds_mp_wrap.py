import warnings
from typing import Optional

import warnings
from typing import Optional

import deepspeed
import torch
from deepspeed.pipe import TiedLayerSpec, LayerSpec
from torch import nn

from general_util.logger import get_child_logger
from models.mpt.modeling_mpt import MPTConfig, SharedEmbedding, MPTBlock, NORM_CLASS_REGISTRY, attn_bias_shape

logger = get_child_logger("MPT.PipelineParallelWrap")


class EmbeddingPipeLayer(nn.Module):
    def __init__(self, config: MPTConfig):
        super().__init__()
        self.config = config

        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        self.embedding_fraction = config.embedding_fraction
        self.is_causal = not self.prefix_lm
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(self.attn_impl, config.n_heads, config.max_seq_len, self.alibi, prefix_lm=self.prefix_lm,
                                               causal=self.is_causal, use_sequence_id=self.attn_uses_sequence_id)

        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        self.wte = SharedEmbedding(config.vocab_size, config.d_model)
        if not self.alibi:
            self.wpe = torch.nn.Embedding(config.max_seq_len, config.d_model, device=config.init_device)
        self.emb_drop = nn.Dropout(config.emb_pdrop)

    @property
    def weight(self):
        return self.wte.weight

    # @torch.no_grad()
    # def _attn_bias(self, device, dtype, attention_mask: Optional[torch.ByteTensor] = None, prefix_mask: Optional[torch.ByteTensor] = None,
    #                sequence_id: Optional[torch.LongTensor] = None):
    #     if not self._attn_bias_initialized:
    #         if self.attn_bias_shape:
    #             self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)
    #             self.attn_bias = build_attn_bias(self.attn_impl, self.attn_bias, self.config.n_heads, self.config.max_seq_len,
    #                                              causal=self.is_causal, alibi=self.alibi, alibi_bias_max=self.alibi_bias_max)
    #         self._attn_bias_initialized = True
    #     if self.attn_impl == 'flash':
    #         return self.attn_bias, attention_mask
    #     if self.attn_bias is not None:
    #         self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)
    #     attn_bias = self.attn_bias
    #     if self.prefix_lm:
    #         assert isinstance(attn_bias, torch.Tensor)
    #         assert isinstance(prefix_mask, torch.Tensor)
    #         attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
    #     if self.attn_uses_sequence_id and sequence_id is not None:
    #         assert isinstance(attn_bias, torch.Tensor)
    #         attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
    #     if attention_mask is not None:
    #         s_k = attention_mask.shape[-1]
    #         if attn_bias is None:
    #             attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
    #         else:
    #             _s_k = max(0, attn_bias.size(-1) - s_k)
    #             attn_bias = attn_bias[:, :, :, _s_k:]
    #         if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
    #             raise ValueError(
    #                 f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
    #         min_val = torch.finfo(attn_bias.dtype).min
    #         attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k).bool(), min_val)
    #     return attn_bias, None

    # @torch.no_grad()
    def _attn_bias(self, attn_bias, device, dtype, attention_mask: Optional[torch.ByteTensor] = None,
                   prefix_mask: Optional[torch.ByteTensor] = None,
                   sequence_id: Optional[torch.LongTensor] = None):
        if self.attn_impl == 'flash':
            return attn_bias, attention_mask
        if attn_bias is not None:
            attn_bias = attn_bias.to(dtype=dtype)

        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)
            assert isinstance(prefix_mask, torch.Tensor)
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(
                    f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            # min_val = torch.finfo(attn_bias.dtype).min
            min_val = torch.finfo(torch.float16).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k).bool(), min_val)
        return attn_bias, None

    def _apply_prefix_mask(self, attn_bias: torch.Tensor, prefix_mask: torch.Tensor):
        (s_k, s_q) = attn_bias.shape[-2:]
        if s_k != self.config.max_seq_len or s_q != self.config.max_seq_len:
            raise ValueError(
                'attn_bias does not match the expected shape. ' +
                f'The last two dimensions should both be {self.config.max_length} ' + f'but are {s_k} and {s_q}.')
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor, sequence_id: torch.LongTensor):
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}')
        attn_bias = attn_bias[..., :seq_len, :seq_len]
        cannot_attend = torch.logical_not(torch.eq(sequence_id.view(-1, seq_len, 1), sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)
        return attn_bias

    def forward(self, args):
        if len(args) == 2:
            input_ids, attention_mask = args
            attn_bias = None
        else:
            input_ids, attention_mask, attn_bias = args

        S = input_ids.size(1)
        assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, ' \
                                             f'this model only supports seq_len<={self.config.max_seq_len}'
        tok_emb = self.wte(input_ids)

        if self.alibi:
            x = tok_emb
        else:
            past_position = 0

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, '
                    f'this model only supports total sequence length <= {self.config.max_seq_len}.')
            pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb

        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
            assert isinstance(self.emb_drop, nn.Module)
            x = self.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(attn_bias, device=x.device, dtype=torch.float32, attention_mask=attention_mask,
                                                    prefix_mask=None, sequence_id=None)

        if attention_mask is None:
            return x, attn_bias

        return x, attn_bias, attention_mask


class ParallelTransformerLayerPipe(MPTBlock):
    def __init__(self, config: MPTConfig, activation_checkpointing: bool = False):
        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        super().__init__(**config.to_dict())

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    if config.verbose:
                        warnings.warn(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

        self.prefix_lm = config.attn_config['prefix_lm']
        self.is_causal = not self.prefix_lm
        self.activation_checkpointing = activation_checkpointing

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        if len(args) == 2:
            x, attn_bias = args
            attention_mask = None
        else:
            x, attn_bias, attention_mask = args

        x, _, _ = MPTBlock.forward(self, x, None, attn_bias, attention_mask, self.is_causal)

        if attention_mask is None:
            return x, attn_bias

        return x, attn_bias, attention_mask

    def _ckpt_forward(self, args):
        if len(args) == 2:
            x, attn_bias = args
            attention_mask = None
        else:
            x, attn_bias, attention_mask = args

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return MPTBlock.forward(module, *inputs)

            return custom_forward

        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            x,
            None,
            attn_bias,
            attention_mask,
            self.is_causal
        )

        if attention_mask is None:
            return outputs[0], attn_bias

        return outputs[0], attn_bias, attention_mask


class LayerNormPipe(nn.Module):
    def __init__(self, config: MPTConfig):
        super().__init__()
        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(
                f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).')
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]

        self.norm_f = norm_class(config.d_model)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    if config.verbose:
                        warnings.warn(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

    def forward(self, args):
        if len(args) == 2:
            x, attn_bias = args
            attention_mask = None
        else:
            x, attn_bias, attention_mask = args

        x = self.norm_f(x)
        return x


class LMHeadPipe(nn.Module):
    def __init__(self, config: MPTConfig):
        super().__init__()
        self.wte = nn.Linear(config.d_model, config.vocab_size, bias=False)

    @property
    def weight(self):
        return self.wte.weight

    def forward(self, args):
        x = args
        logits = nn.functional.linear(x, self.wte.weight)
        return logits


def get_layers_from_config(model_config: MPTConfig, activation_checkpointing: bool = False):
    layers = [
        TiedLayerSpec("wte", EmbeddingPipeLayer, model_config, tied_weight_attr="weight"),
        *[LayerSpec(ParallelTransformerLayerPipe, model_config, activation_checkpointing)
          for _ in range(model_config.n_layers)],
        LayerSpec(LayerNormPipe, model_config),
        TiedLayerSpec("wte", LMHeadPipe, model_config, tied_weight_attr="weight"),
    ]
    return layers

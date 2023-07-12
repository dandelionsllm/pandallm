import os

import deepspeed
import torch
from deepspeed.pipe import TiedLayerSpec, LayerSpec
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        input_ids, attention_mask, position_ids = ipt
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds, attention_mask, position_ids


class LlamaPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM, layer_idx):
        super().__init__()
        self.layer = model.model.layers[layer_idx]
        self.gradient_checkpointing = model.model.gradient_checkpointing

    def forward(self, ipt):
        hidden_states, attention_mask, position_ids = ipt

        if self.gradient_checkpointing and self.training:
            output_attentions = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            # layer_outputs = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(self.layer),
            #     hidden_states,
            #     attention_mask,
            #     position_ids,
            #     None,
            # )
            # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
            outputs = deepspeed.checkpointing.checkpoint(
                create_custom_forward(self.layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
            layer_outputs = [outputs]
        else:
            layer_outputs = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                # past_key_value=past_key_value,
                # output_attentions=output_attentions,
                # use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]
        return hidden_states, attention_mask, position_ids


class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.norm = model.model.norm

    def forward(self, ipt):
        hidden_states, attention_mask, position_ids = ipt
        hidden_states = self.norm(hidden_states)

        return hidden_states


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.lm_head = model.lm_head
        self.weight = self.lm_head.weight
        self.config = model.config

    def forward(self, ipt):
        hidden_states = ipt
        logits = torch.nn.functional.linear(hidden_states, self.lm_head.weight)

        return logits


def loss_fn(outputs, labels):
    logits = outputs
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    return loss


class LossFNRatio:
    def __init__(self, ignore_index: int = -100, merit_ratio: float = 0.5):
        self.ignore_index = ignore_index
        self.merit_ratio = merit_ratio
        logger.info(f"LossFNRatio: ignore_index={ignore_index}, merit_ratio={merit_ratio}")

    def __call__(self, outputs, labels):
        logits = outputs

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        bsz, seq_len = shift_labels.shape
        sub_bsz = bsz // 2

        loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)

        sub_logits0 = shift_logits[:sub_bsz]
        sub_logits1 = shift_logits[sub_bsz:]
        sub_labels0 = shift_labels[:sub_bsz]
        sub_labels1 = shift_labels[sub_bsz:]
        loss0 = loss_fct(sub_logits0.reshape(-1, sub_logits0.size(-1)), sub_labels0.reshape(-1))
        loss1 = loss_fct(sub_logits1.reshape(-1, sub_logits1.size(-1)), sub_labels1.reshape(-1))
        loss = (1 - self.merit_ratio) * loss0 + self.merit_ratio * loss1
        return loss


def get_model(model):
    layers = [TiedLayerSpec("weight", EmbeddingPipeLayer, model=model, tied_weight_attr="weight"),
              *[LayerSpec(LlamaPipeLayer, model=model, layer_idx=idx) for idx in range(model.config.num_hidden_layers)],
              LayerSpec(FLNPipeLayer, model=model),
              TiedLayerSpec("weight", LMPipeLayer, model=model, tied_weight_attr="weight"),
              ]
    return layers


class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, attention_mask, position_ids = args
        inputs_embeds = super().forward(input_ids)
        return inputs_embeds, attention_mask, position_ids


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, activation_checkpointing: bool = False):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing
        # for name, param in self.named_parameters():
        #     if "norm" in name:
        #         continue
        #     param.data = param.data.to(dtype)

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, attention_mask, position_ids = args
        outputs = LlamaDecoderLayer.forward(self,
                                            hidden_states,
                                            attention_mask,
                                            position_ids,
                                            )
        return outputs[0], attention_mask, position_ids

    def _ckpt_forward(self, args):
        hidden_states, attention_mask, position_ids = args

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return LlamaDecoderLayer.forward(module, *inputs)

            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
            None,
        )
        # layer_outputs = torch.utils.checkpoint.checkpoint(
        #     create_custom_forward(self),
        #     hidden_states,
        #     attention_mask,
        #     position_ids,
        #     None,
        # )

        return outputs, attention_mask, position_ids


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, attention_mask, position_ids = args
        last_hidden_states = super().forward(hidden_states)
        return last_hidden_states


class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        hidden_states = args
        logits = super().forward(hidden_states)
        return logits


class LossLayer(torch.nn.Module):
    def forward(self, args):
        logits, labels = args
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss


def get_layers_from_config(model_config, activation_checkpointing: bool = False):
    """
    `tie_word_embeddings` in LLaMA is set to `false`.
    """
    layers = [
        LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
        # TiedLayerSpec("weight", EmbeddingPipe, model_config.vocab_size, model_config.hidden_size, tied_weight_attr="weight"),
        *[LayerSpec(ParallelTransformerLayerPipe, model_config, activation_checkpointing)
          for _ in range(model_config.num_hidden_layers)],
        LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
        LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False),
        # TiedLayerSpec("weight", LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False,
        #               tied_weight_attr="weight"),
        # LayerSpec(LossLayer),
    ]
    return layers

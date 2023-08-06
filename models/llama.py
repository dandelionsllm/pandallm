import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Callable

import bitsandbytes as bnb
import hydra.utils
import omegaconf
import torch
import math
from einops import rearrange
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaConfig, SequenceClassifierOutputWithPast, \
    LlamaDecoderLayer, LlamaForCausalLM, apply_rotary_pos_emb, repeat_kv

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import fold_tensor, get_accuracy

logger = get_child_logger(__name__)

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

PAD_TOKEN_ID = 32000


def llama_fast_attention_wrap(attn_layer: nn.Module, vanilla_torch: bool = False, var_len: bool = False):
    self = attn_layer

    if not vanilla_torch:
        from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_kvpacked_func

        if var_len:  # To be honest, I do not know when to use this function.
            from flash_attn.flash_attn_interface import flash_attn_varlen_func
            from deepspeed.accelerator import get_accelerator

    def _forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if vanilla_torch:
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False
            ):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True,
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            if var_len:
                # Copied from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py#L400-L429
                if get_accelerator().device_name() == 'cuda':
                    # goes for cuda device
                    query_states, key_states, value_states = [rearrange(x, 'b h s d -> (b s) h d')
                                                              for x in [query_states, key_states, value_states]]
                    cu_seqlens_q = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                                device=query_states.device)
                else:
                    # goes for other device
                    # q, k, v = [rearrange(x, 'b s h d -> b h s d').contiguous() for x in [q, k, v]]
                    pass

                if self.training:
                    # during training q,k,v always have same seqlen
                    assert q_len == kv_seq_len

                    is_causal = True
                    cu_seqlens_k = cu_seqlens_q if get_accelerator().device_name() == 'cuda' else None
                else:
                    # turn off FA causal mask after first inference autoregressive iteration
                    # only on first autoregressive step q,k,v have same seqlen
                    is_causal = q_len == kv_seq_len
                    cu_seqlens_k = torch.arange(0, (bsz + 1) * kv_seq_len, step=kv_seq_len, dtype=torch.int32,
                                                device=query_states.device) if get_accelerator().device_name() == 'cuda' else None
                    # self.dropout_p = 0

                attn_output = flash_attn_varlen_func(
                    query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, q_len, kv_seq_len,
                    # self.dropout_p,
                    # softmax_scale=self.softmax_scale,
                    causal=is_causal
                )  # if get_accelerator().device_name() == 'cuda' else flash_attn_builder.flash_attn_func(
                # q, k, v, self.dropout_p, self.softmax_scale, is_causal
                # )
                attn_output = rearrange(attn_output, '(b s) ... -> b s ...', b=bsz) if get_accelerator().device_name() == 'cuda' \
                    else rearrange(attn_output, 'b h s d -> b s h d').contiguous()
            elif q_len == kv_seq_len:
                # repeat k/v heads if n_kv_heads < n_heads
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                # transform the data into the format required by flash attention
                qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
                qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

                attn_output = flash_attn_qkvpacked_func(qkv, causal=True)  # [bsz, q_len, nh, hd]
            else:
                kv = torch.stack([key_states, value_states], dim=2)  # [bsz, nh, 2, kv_seq_len, hd]
                kv = kv.transpose(1, 3)  # [bsz, kv_seq_len, 2, nh, hd]
                query_states = query_states.transpose(1, 2)  # [bsz, nh, q_len, hd]

                attn_output = flash_attn_kvpacked_func(query_states, kv, causal=True)  # [bsz, q_len, nh, hd]

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        assert not output_attentions
        # if not output_attentions:
        #     attn_weights = None

        return attn_output, None, past_key_value

    self.forward = _forward


def wrap_causal_lm_w_flash_attention(pretrained_model_name_or_path,
                                     enable_flash_attention: bool = False,
                                     flash_attention_vanilla_torch: bool = False,
                                     flash_attention_var_len: bool = False,
                                     **kwargs):
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if enable_flash_attention:
        logger.info("⚡⚡⚡ enable llama flash attention.")

        for layer in model.model.layers:
            llama_fast_attention_wrap(layer.self_attn, vanilla_torch=flash_attention_vanilla_torch, var_len=flash_attention_var_len)

    return model


def deepspeed_inference_policy():
    injection_policy = {LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
    return injection_policy


def find_all_linear_names(model, bits: int, add_lm_head: bool = False):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names.add("lm_head")

    if 'lm_head' in lora_module_names and not add_lm_head:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def return_single_device_map():
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}


def return_cpu_device_map():
    return {"": "cpu"}


def return_single_device_map_emb():
    return {"embed_tokens": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0)), "": "cpu"}


def embedding_tuner_llama_strategy(tunable_layer_norm: bool = False):
    tunable_param_names = ["embed_tokens", "lm_head"]
    if tunable_layer_norm:
        tunable_param_names.append("norm")

    def _call_(model: nn.Module):
        for name, param in model.named_parameters():
            if all([x not in name for x in tunable_param_names]):
                param.requires_grad = False

    return _call_


@dataclass
class MultipleChoicePreTrainModelOutput(SequenceClassifierOutputWithPast):
    mlm_loss: torch.FloatTensor = None
    mlm_acc: torch.FloatTensor = None
    cls_loss: torch.FloatTensor = None
    cls_acc: torch.FloatTensor = None
    pair_loss: torch.FloatTensor = None
    tagging_loss: torch.FloatTensor = None
    path_gen_loss: torch.FloatTensor = None
    path_gen_acc: torch.FloatTensor = None
    ent_gen_loss: torch.FloatTensor = None
    ent_gen_acc: torch.FloatTensor = None
    rel_ctr_loss: torch.FloatTensor = None
    local_ctr_loss: torch.FloatTensor = None
    local_ctr_acc: torch.FloatTensor = None
    original_logits: torch.FloatTensor = None


class LlamaPreTrainedModelPeftMixin(LlamaPreTrainedModel, ABC):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], freeze_strategy: Optional[Callable] = None,
                        *model_args, **kwargs):
        if "vocab_size" in kwargs and "pad_token_id" in kwargs:
            # Hack here to avoid embedding weight size mismatch during loading pre-trained weights.
            vocab_size = kwargs.pop("vocab_size")
            pad_token_id = kwargs.pop("pad_token_id")
        else:
            vocab_size = None
            pad_token_id = None

        enable_flash_attention = kwargs.pop("enable_flash_attention", False)
        flash_attention_vanilla_torch = kwargs.pop("flash_attention_vanilla_torch", False)
        flash_attention_var_len = kwargs.pop("flash_attention_var_len", False)

        use_peft = kwargs.pop("use_peft", False)
        lora_config = kwargs.pop("lora_config", None)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if vocab_size is not None and pad_token_id is not None:
            # assert vocab_size == model.config.vocab_size + 1, "Currently, only hack here to add pad token id is supported. "
            if vocab_size != model.config.vocab_size:
                model.resize_token_embeddings(vocab_size)
            model.config.pad_token_id = pad_token_id

        if use_peft:
            if lora_config is None:
                lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

            # logger.info(*model_args)
            # logger.info(kwargs)
            logger.info(lora_config)
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

            if vocab_size is not None and pad_token_id is not None:
                # assert vocab_size == model.config.vocab_size + 1, "Currently, only hack here to add pad token id is supported. "
                if vocab_size != model.config.vocab_size:
                    model.resize_token_embeddings(vocab_size)
                model.config.pad_token_id = pad_token_id

            logger.info(f"LORA Config: {lora_config}")
            logger.info(lora_config.target_modules.__class__)
            if isinstance(lora_config.target_modules, omegaconf.listconfig.ListConfig):
                lora_config.target_modules = list(lora_config.target_modules)
            elif isinstance(lora_config.target_modules, omegaconf.DictConfig):
                lora_config.target_modules = hydra.utils.instantiate(lora_config.target_modules, model=model)
            else:
                raise ValueError(f"Unsupported type of target modules: {lora_config.target_modules.__class__}")

            logger.info(lora_config.target_modules.__class__)
            logger.info(lora_config.target_modules)
            gradient_checkpointing = model.model.gradient_checkpointing
            if load_in_8bit or load_in_4bit:
                # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=gradient_checkpointing)
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
            model = get_peft_model(model, lora_config)

            compute_dtype = kwargs["torch_dtype"]
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if compute_dtype == torch.bfloat16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if compute_dtype and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

            model.print_trainable_parameters()

        if enable_flash_attention:
            logger.info("⚡⚡⚡ enable llama flash attention.")

            for layer in model.model.layers:
                llama_fast_attention_wrap(layer.self_attn, vanilla_torch=flash_attention_vanilla_torch, var_len=flash_attention_var_len)

        if freeze_strategy is not None:
            freeze_strategy(model)

        logger.info(f"Config pad token id after loading pre-trained weights: {model.config.pad_token_id}")

        return model

    @classmethod
    def from_pretrained_peft_eval(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        base_model_name_or_path = kwargs.pop("base_model_name_or_path", pretrained_model_name_or_path)

        model = super().from_pretrained(base_model_name_or_path, *model_args, **kwargs)
        model = PeftModel.from_pretrained(model, pretrained_model_name_or_path, *model_args, **kwargs)
        return model


class LlamaForMultipleChoiceCLS(LlamaPreTrainedModelPeftMixin, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        logger.info(f"Config pad token id: {self.config.pad_token_id}")

        # Initialize weights and apply final processing
        self.post_init()

        self.init_metric("loss", "acc")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_cls_head(self):
        return self.score

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]
        input_ids = fold_tensor(input_ids)
        attention_mask = fold_tensor(attention_mask)

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = torch.zeros(batch_size, 1).fill_(-1).to(hidden_states.device)
        else:
            sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1, keepdim=True) - 1).to(hidden_states.device)
        length_index = sequence_lengths.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)).contiguous()

        sentence_representation = torch.gather(hidden_states, 1, length_index).squeeze(1)
        reshaped_logits = self.score(sentence_representation).view(-1, num_choices)

        loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if not self.training:
                acc, true_label_num = get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            cls_loss=cls_loss,
            logits=reshaped_logits,
        )


class LlamaForMultipleChoiceCausalLM(LlamaPreTrainedModelPeftMixin, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: LlamaConfig, add_lm_loss: bool = False):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        logger.info(f"Config pad token id: {self.config.pad_token_id}")

        # Initialize weights and apply final processing
        self.post_init()

        self.add_lm_loss = add_lm_loss

        metrics = ["loss", "acc", "cls_loss"]
        if add_lm_loss:
            metrics.extend(["mlm_loss"])
        self.init_metric(*metrics)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        input_ids = fold_tensor(input_ids)
        attention_mask = fold_tensor(attention_mask)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        shifted_logits = logits[..., :-1, :].contiguous()

        label_mask = input_ids.ne(self.config.pad_token_id)
        # logger.info(label_mask[0])
        # if input_lens is not None:
        # keep only logits after the end of the condition part in each item of the batch
        # [batch_size * num_choices, input_lens]
        if input_lens is not None:
            lens_mask = torch.arange(input_ids.size(1), device=label_mask.device)[None, :] >= input_lens[:, None]
            # logger.info(lens_mask[0])
            label_mask = label_mask & lens_mask
        # logger.info(label_mask[0])
        lm_labels = input_ids.masked_fill(~label_mask, -100).contiguous()
        shifted_lm_labels = lm_labels[..., 1:].contiguous()

        lm_ppl = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(shifted_logits.view(-1, logits.size(-1)),
                                                                          shifted_lm_labels.view(-1))
        lm_ppl = lm_ppl.reshape(batch_size, num_choices, input_ids.size(1) - 1)
        # [batch_size, num_choices]
        true_seq_len = label_mask.to(dtype=lm_ppl.dtype).reshape(batch_size, num_choices, -1).sum(-1).detach()
        # logger.info(true_seq_len.sum().item())
        no_seq_mask = true_seq_len.eq(0)
        lm_ppl = lm_ppl.sum(-1) / (true_seq_len + no_seq_mask.to(dtype=lm_ppl.dtype))
        assert lm_ppl.size() == (batch_size, num_choices)

        reshaped_logits = -lm_ppl
        reshaped_logits = reshaped_logits.masked_fill(no_seq_mask, -10000.0)

        loss = 0.
        cls_loss = 0.
        lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            cls_label_mask = torch.gather(no_seq_mask, 1, labels[:, None]).squeeze(1)
            masked_labels = labels.masked_fill(cls_label_mask, -1)
            cls_loss = loss_fct(reshaped_logits, masked_labels)
            if masked_labels.ne(-1).sum().item() > 0:
                loss = loss + cls_loss
            else:
                cls_loss = 0.

            if self.add_lm_loss:
                # lm_loss = lm_ppl[:, 0].mean()
                masked_lm_ppl = lm_ppl.masked_fill(no_seq_mask, 0.0)
                lm_loss = torch.gather(masked_lm_ppl, 1, labels[:, None]).squeeze(1).mean()
                loss = loss + lm_loss

            if not self.training:
                acc, true_label_num = get_accuracy(reshaped_logits, masked_labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss, n=true_label_num)

                if self.add_lm_loss:
                    self.eval_metrics.update("mlm_loss", val=lm_loss, n=1)

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            cls_loss=cls_loss,
            mlm_loss=lm_loss,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class LlamaForConditionalGeneration(LlamaPreTrainedModelPeftMixin, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: LlamaConfig, gradient_checkpointing=False):
        super().__init__(config)
        self.model = LlamaModel(config)
        # set gradient checkpointing
        # self.model.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.config.use_cache = False
            self.gradient_checkpointing_enable()
        logger.info(f"gradient_checkpointing: {gradient_checkpointing}")

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        metrics = ["loss", "acc"]
        self.init_metric(*metrics)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        shifted_logits = logits[..., :-1, :].contiguous()

        label_mask = input_ids.ne(self.config.pad_token_id)
        # logger.info(label_mask[0])
        # if input_lens is not None:
        # keep only logits after the end of the condition part in each item of the batch
        # [batch_size * num_choices, input_lens]
        if input_lens is not None:
            lens_mask = torch.arange(input_ids.size(1), device=label_mask.device)[None, :] >= input_lens[:, None]
            # logger.info(lens_mask[0])
            label_mask = label_mask & lens_mask
        # logger.info(label_mask[0])
        lm_labels = input_ids.masked_fill(~label_mask, -1).contiguous()
        shifted_lm_labels = lm_labels[..., 1:].contiguous()

        if shifted_logits.device != shifted_lm_labels.device:
            shifted_logits = shifted_logits.to(shifted_lm_labels.device)

        # loss = 0.
        # if labels is not None:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        lm_loss = loss_fct(shifted_logits.view(-1, logits.size(-1)), shifted_lm_labels.view(-1))
        loss = lm_loss
        # logger.info(loss)

        if not self.training:
            acc, true_label_num = get_accuracy(shifted_logits, shifted_lm_labels)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

            score_loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            score_loss = score_loss_fct(shifted_logits.view(-1, logits.size(-1)), shifted_lm_labels.view(-1))
            score_loss = score_loss.reshape(batch_size, -1)
            score_loss = score_loss.sum(dim=-1) / label_mask.sum(dim=-1).float()
            return MultipleChoicePreTrainModelOutput(
                loss=loss,
                logits=-score_loss,
            )
        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=shifted_logits,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class LlamaForConditionalGenerationFlan(LlamaForConditionalGeneration, LogMixin, ABC):
    def __init__(self, config: LlamaConfig, gradient_checkpointing=False, merit_ratio: float = 0.5):
        super().__init__(config, gradient_checkpointing)
        self.merit_ratio = merit_ratio

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            flan_input_ids: Optional[torch.LongTensor] = None,
            flan_attention_mask: Optional[torch.FloatTensor] = None,
            flan_token_type_ids: Optional[torch.LongTensor] = None,
            flan_input_lens: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs1 = super().forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   input_lens=input_lens,
                                   return_dict=return_dict)
        outputs2 = super().forward(input_ids=flan_input_ids,
                                   attention_mask=flan_attention_mask,
                                   input_lens=flan_input_lens,
                                   return_dict=return_dict)
        # if torch.isnan(outputs1.loss):
        #     print("Normal inputs NAN loss")
        # if torch.isnan(outputs2.loss):
        #     print("Flan inputs NAN loss")

        # loss = (outputs1.loss + outputs2.loss) / 2
        loss = self.merit_ratio * outputs1.loss + (1 - self.merit_ratio) * outputs2.loss

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            mlm_loss=outputs1.loss,
            logits=outputs1.logits,
        )


def mask_according_lens(input_ids, input_lens, pad_token_id):
    label_mask = input_ids.ne(pad_token_id)
    # keep only logits after the end of the condition part in each item of the batch
    # [batch_size * num_choices, input_lens]
    lens_mask = torch.arange(input_ids.size(1), device=label_mask.device)[None, :] >= input_lens[:, None]
    label_mask = label_mask & lens_mask
    return label_mask


def token_wise_ctr_forward(
        model: LlamaModel,
        linear_layer: nn.Module,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_lens: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
):
    batch_size, num_choice = input_ids.size()[:2]
    input_ids = fold_tensor(input_ids)
    attention_mask = fold_tensor(attention_mask)

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True)

    hidden_states = outputs[0]
    label_mask = mask_according_lens(input_ids, input_lens, pad_token_id)
    # [batch_size * num_choices, seq_len]
    token_logits = linear_layer(hidden_states).squeeze(-1)
    logits = token_logits.masked_fill(~label_mask, 0).sum(dim=1) / label_mask.sum(dim=1)
    logits = logits.view(batch_size, num_choice)
    return logits


class LlamaCtrAndLMPretrain(LlamaForConditionalGeneration, ABC):
    def __init__(self, config: LlamaConfig, gradient_checkpointing=False):
        super().__init__(config, gradient_checkpointing)

        self.linear = nn.Linear(config.hidden_size, 1)
        self.init_metric("loss", "acc", "cls_loss")

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            input_lens: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            flan_input_ids: Optional[torch.LongTensor] = None,
            flan_attention_mask: Optional[torch.FloatTensor] = None,
            flan_token_type_ids: Optional[torch.LongTensor] = None,
            flan_input_lens: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoicePreTrainModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ctr_logits = token_wise_ctr_forward(self.model, self.linear, input_ids, attention_mask, input_lens, self.config.pad_token_id)
        ctr_loss = nn.CrossEntropyLoss()(ctr_logits, labels)

        lm_outputs = super().forward(input_ids=flan_input_ids,
                                     attention_mask=flan_attention_mask,
                                     input_lens=flan_input_lens,
                                     return_dict=return_dict)
        lm_loss = lm_outputs.loss
        loss = ctr_loss + lm_loss

        ctr_acc = get_accuracy(ctr_logits, labels)
        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=ctr_logits,
            mlm_loss=lm_loss,
            cls_loss=ctr_loss,
            cls_acc=ctr_acc,
        )

import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import omegaconf
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_int8_training
)
from torch import nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel, LlamaConfig, SequenceClassifierOutputWithPast, \
    LlamaDecoderLayer

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules.layers import fold_tensor, get_accuracy

logger = get_child_logger(__name__)

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

PAD_TOKEN_ID = 32000


def deepspeed_inference_policy():
    injection_policy = {LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
    return injection_policy


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


class LlamaPreTrainedModelPeftMixin(LlamaPreTrainedModel, ABC):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if "vocab_size" in kwargs and "pad_token_id" in kwargs:
            # Hack here to avoid embedding weight size mismatch during loading pre-trained weights.
            vocab_size = kwargs.pop("vocab_size")
            pad_token_id = kwargs.pop("pad_token_id")
        else:
            vocab_size = None
            pad_token_id = None

        use_peft = kwargs.pop("use_peft", False)
        if not use_peft:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            lora_config = kwargs.pop("lora_config", None)
            if lora_config is None:
                lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

            logger.info(f"LORA Config: {lora_config}")
            logger.info(lora_config.target_modules.__class__)
            if isinstance(lora_config.target_modules, omegaconf.listconfig.ListConfig):
                lora_config.target_modules = list(lora_config.target_modules)
            logger.info(lora_config.target_modules.__class__)

            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

            load_in_8bit = kwargs.pop("load_in_8bit", False)
            if load_in_8bit:
                model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False)
            model = get_peft_model(model, lora_config)

            if hasattr(model, "get_cls_head"):
                for param in model.get_cls_head().parameters():
                    param.requires_grad = True

            model.print_trainable_parameters()

        if vocab_size is not None and pad_token_id is not None:
            assert vocab_size == model.config.vocab_size + 1, "Currently, only hack here to add pad token id is supported. "
            model.resize_token_embeddings(vocab_size)
            model.config.pad_token_id = pad_token_id

        logger.info(f"Config pad token id after loading pre-trained weights: {model.config.pad_token_id}")

        return model

    @classmethod
    def from_pretrained_peft_eval(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        base_model_name_or_path = kwargs.pop("base_model_name_or_path", pretrained_model_name_or_path)

        model = super().from_pretrained(base_model_name_or_path, *model_args, **kwargs)
        model = PeftModel.from_pretrained(model, pretrained_model_name_or_path)
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

    def __init__(self, config: LlamaConfig, add_lm_loss: bool = False, add_pad_token_id: bool = False):
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
        self.model.gradient_checkpointing = gradient_checkpointing
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

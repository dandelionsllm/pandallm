from abc import ABC
from typing import Optional, Tuple, Union, Dict

import torch
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5Stack,
    Seq2SeqLMOutput, T5ForConditionalGeneration, BaseModelOutput
)
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("T5")


class T5ForSeq2Seq(T5ForConditionalGeneration, LogMixin, ABC):

    def __init__(self, config: T5Config, tokenizer: str):
        super().__init__(config)
        self.config = config

        self.init_metric("loss", "acc", "bleu")

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer, use_fast=False)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Dict[str, Tensor], T5Stack]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.

        Returns:

        Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> import torch

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", return_tensors="pt")).unsqueeze(0)  # Batch size 1
        >>> outputs = model.generate(input_ids)

        >>> print("Generated: {}".format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        Generated: Hello, my dog is cute
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            label_padding_mask = labels == self.config.pad_token_id
            labels[label_padding_mask] = -1
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom):
            #  Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if not self.training:
                # Generate sentences for BLEU evaluation
                max_output_length = labels.size(1)
                # Greedy decoding.
                eval_gen_sentences = self.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_output_length,
                                                   num_beams=1, do_sample=False)
                eval_gen_sentences = self.tokenizer.batch_decode(eval_gen_sentences, skip_special_tokens=True)
                target = self.tokenizer.batch_decode(labels.masked_fill(label_padding_mask, self.config.pad_token_id),
                                                     skip_special_tokens=True)
                # bleu = sum(
                #     [sentence_bleu([tgt.split()], gen_sentence.split()) for tgt, gen_sentence in zip(target, eval_gen_sentences)]
                # ) / labels.size(0)
                bleu = sum(
                    [sentence_bleu([word_tokenize(tgt)], word_tokenize(gen_sentence))
                     for tgt, gen_sentence in zip(target, eval_gen_sentences)]
                )
                self.eval_metrics.update("bleu", bleu, n=labels.size(0))

                acc, true_label_num = layers.get_accuracy(lm_logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

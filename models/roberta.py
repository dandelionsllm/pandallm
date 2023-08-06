import copy
from abc import ABC
from dataclasses import dataclass

import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig, RobertaLMHead, \
    MaskedLMOutput, SequenceClassifierOutput, RobertaEncoder
from transformers.models.t5.modeling_t5 import T5Stack, T5Config

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from modules import layers

logger = get_child_logger("RoBERTa")


class RobertaForMultipleChoice(RobertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 freeze_encoder: bool = False,
                 no_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.classifier = nn.Linear(config.hidden_size, 1)

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.no_pooler = no_pooler
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            layers.freeze_module(self.roberta)

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            op_mask: Tensor = None,
            labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.no_pooler:
            pooled_output = outputs[0][:, 0]
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class MultipleChoicePreTrainModelOutput(MultipleChoiceModelOutput):
    mlm_loss: torch.FloatTensor = None
    mlm_acc: torch.FloatTensor = None
    cls_loss: torch.FloatTensor = None
    cls_acc: torch.FloatTensor = None


@dataclass
class SequenceClassificationPreTrainModelOutput(SequenceClassifierOutput):
    mlm_loss: torch.FloatTensor = None
    mlm_acc: torch.FloatTensor = None
    cls_loss: torch.FloatTensor = None
    cls_acc: torch.FloatTensor = None


class RobertaForMultipleChoiceForPreTrain(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                if mlm_attention_mask is None:
                    mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return MultipleChoicePreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
        )


class RobertaForSequenceClassificationForPreTrain(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 mlm_alpha: float = 1.0,
                 mlm_disabled: bool = False,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.vocab_size = config.vocab_size

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 2)  # ``1`` for ``true`` and ``0`` for ``false``

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc", "mlm_loss", "mlm_acc", "cls_loss")

        self.mlm_alpha = mlm_alpha
        # The option is added to disable the MLM loss but keep computing the MLM accuracy on the validation set,
        # in order to observe if there is the catastrophic forgetting problem.
        self.mlm_disabled = mlm_disabled

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            mlm_input_ids: Tensor = None,
            mlm_attention_mask: Tensor = None,
            mlm_labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, self.config.num_labels)

        loss = 0.
        mlm_loss = 0.
        cls_loss = 0.
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = loss + cls_loss

            if mlm_labels is not None and (self.mlm_disabled is False or self.training is False):
                # if mlm_attention_mask is None:
                #     mlm_attention_mask = attention_mask.reshape(reshaped_logits.size(0), num_choices, -1)[:, 0]

                mlm_outputs = self.roberta(
                    mlm_input_ids,
                    attention_mask=mlm_attention_mask,
                    return_dict=return_dict
                )

                mlm_scores = self.lm_head(mlm_outputs[0])
                mlm_loss = self.mlm_alpha * loss_fct(mlm_scores.reshape(-1, self.vocab_size), mlm_labels.reshape(-1))
                if not self.mlm_disabled:
                    loss = loss + mlm_loss
            else:
                mlm_scores = None
                mlm_loss = None

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)
                self.eval_metrics.update("cls_loss", val=cls_loss.item(), n=true_label_num)

                if mlm_labels is not None:
                    acc, true_label_num = layers.get_accuracy(mlm_scores, mlm_labels)
                    self.eval_metrics.update("mlm_acc", val=acc, n=true_label_num)
                    self.eval_metrics.update("mlm_loss", val=mlm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:] + (mlm_loss, cls_loss,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassificationPreTrainModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mlm_loss=mlm_loss,
            cls_loss=cls_loss,
        )


class RobertaForMaskedLM(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]

    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.vocab_size = config.vocab_size

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))

            if not self.training:
                acc, true_label_num = layers.get_accuracy(prediction_scores, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", masked_lm_loss.item(), n=true_label_num)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceInMLM(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]

    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.vocab_size = config.vocab_size

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.init_weights()

        self.init_metric("loss", "acc")

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            mlm_labels=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        seq_len = input_ids.shape[1]
        batch_size = labels.size(0)

        loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")
        masked_lm_loss = -loss_fct(prediction_scores.view(-1, self.vocab_size), mlm_labels.view(-1)).reshape(-1, seq_len)

        # batch_size * choice_num
        logits = masked_lm_loss.sum(dim=-1) / (mlm_labels != -1).sum(dim=-1)
        logits = logits.reshape(batch_size, -1)

        loss = None
        if labels is not None:

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", acc, n=true_label_num)
                self.eval_metrics.update("loss", loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoiceForZeroShot(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 freeze_encoder: bool = False,
                 freeze_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu)

        self.freeze_pooler = freeze_pooler
        if self.freeze_pooler:
            layers.freeze_module(self.pooler)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            layers.freeze_module(self.roberta)

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoicePrompt(RobertaPreTrainedModel, LogMixin, ABC):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: RobertaConfig,
                 mlp_hidden_size: int = 768,
                 prompt_mlp_hidden_size: int = 768,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 fs_checkpoint_start_layer_id: int = 0,
                 freeze_encoder: bool = False,
                 freeze_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.prompt_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, prompt_mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(prompt_mlp_hidden_size, config.hidden_size),
        )

        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh()
        )
        self.cls = nn.Linear(mlp_hidden_size, 1)

        if fs_checkpoint:
            for i in range(fs_checkpoint_start_layer_id, config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.freeze_pooler = freeze_pooler
        if self.freeze_pooler:
            for param in self.pooler.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for name, param in self.roberta.named_parameters():
                if 'embeddings.word_embeddings' not in name:
                    param.requires_grad = False

        self.init_weights()

        self.init_metric("loss", "acc")

    @staticmethod
    def fold_tensor(x: Tensor):
        if x is None:
            return x
        return x.reshape(-1, x.size(-1))

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            prefix_pos: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)
        prefix_pos = self.fold_tensor(prefix_pos)

        embed_layer = self.roberta.embeddings.word_embeddings
        input_embeds = embed_layer(input_ids)

        ex_prefix_pos = prefix_pos.unsqueeze(-1).expand(-1, -1, input_embeds.size(-1))
        prefix_embed = torch.gather(input_embeds, index=ex_prefix_pos, dim=1)
        prefix_embed = self.prompt_mlp(prefix_embed)
        input_embeds = torch.scatter(input_embeds, dim=1, index=ex_prefix_pos, src=prefix_embed.to(dtype=input_embeds.dtype))

        if self.freeze_encoder:
            input_embeds = layers.keep_grad_prompt(input_embeds, prefix_pos)

        outputs = self.roberta(
            inputs_embeds=input_embeds,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0][:, 0]

        logits = self.cls(self.dropout(self.pooler(pooled_output)))
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(reshaped_logits, labels)
            loss = cls_loss

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForSequenceClassification(RobertaPreTrainedModel, LogMixin, ABC):
    def __init__(self, config: RobertaConfig,
                 fs_checkpoint: bool = False,
                 fs_checkpoint_offload_to_cpu: bool = False,
                 fs_checkpoint_maintain_forward_counter: bool = False,
                 freeze_encoder: bool = False,
                 no_pooler: bool = False):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=getattr(config, "pooler_dropout", config.hidden_dropout_prob))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if fs_checkpoint:
            for i in range(config.num_hidden_layers):
                self.roberta.encoder.layer[i] = checkpoint_wrapper(self.roberta.encoder.layer[i],
                                                                   offload_to_cpu=fs_checkpoint_offload_to_cpu,
                                                                   maintain_forward_counter=fs_checkpoint_maintain_forward_counter)

        self.no_pooler = no_pooler
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            layers.freeze_module(self.roberta)

        self.init_weights()

        self.init_metric("loss", "acc")

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,
            token_type_ids: Tensor = None,
            labels: Tensor = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.no_pooler:
            pooled_output = outputs[0][:, 0]
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMultipleChoicePath(RobertaForMultipleChoice, ABC):
    def __init__(self, config: RobertaConfig, num_decoder_layers):
        super().__init__(config)

        self.t5_config = T5Config()
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.num_layers = num_decoder_layers

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.decoder = T5Stack(self.t5_config)
        self.decoder.post_init()

        self.c = nn.Parameter(torch.Tensor(1, 1, 1, self.t5_config.d_model))
        self.c.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.register_buffer("embed_pad", torch.zeros(1, 1, 1, self.t5_config.d_model))

        self.weight_q = nn.Linear(self.t5_config.d_model, self.t5_config.d_model, bias=False)
        self.weight_k = nn.Linear(self.t5_config.d_model, self.t5_config.d_model, bias=False)
        self.weight_o = nn.Linear(self.t5_config.d_model, self.t5_config.d_model, bias=False)

        self.classifier = nn.Linear(self.t5_config.d_model, 1)

        self.model_parallel = False
        self.device_map = None

        self.init_weights()

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                op_mask: Tensor = None,
                labels: Tensor = None,
                part_index: Tensor = None,
                part_token_mask: Tensor = None,
                part_occur_mask: Tensor = None,
                part_mask: Tensor = None,
                pos_index: Tensor = None,
                pos_token_mask: Tensor = None,
                pos_occur_mask: Tensor = None,
                pos_mask: Tensor = None,
                part_decoder_input_ids: Tensor = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                use_cache=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.t5_config.use_cache
        num_choices = input_ids.shape[1]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_outputs = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1))

        part_hidden = self.parse_span_rep(seq_outputs, part_index, part_token_mask, part_occur_mask)
        pos_hidden = self.parse_span_rep(seq_outputs, pos_index, pos_token_mask, pos_occur_mask)

        sample_num = part_decoder_input_ids.size(2)
        part_num = part_mask.size(2)
        part_decoder_input_ids = part_decoder_input_ids.reshape(batch_size, num_choices, sample_num * part_num)

        part_decoder_input_embeds = torch.gather(part_hidden, dim=2,
                                                 index=part_decoder_input_ids.unsqueeze(-1).expand(-1, -1, -1, seq_outputs.size(-1)))
        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size, num_choices * sample_num, part_num, -1)

        embed_pad = self.embed_pad.expand(batch_size, num_choices * sample_num, -1, -1)
        part_decoder_input_embeds = torch.cat([part_decoder_input_embeds, embed_pad], dim=2)

        part_c_index = part_mask.sum(dim=2)[:, :, None, None, None].expand(
            -1, -1, sample_num, 1, seq_outputs.size(-1)).reshape(batch_size, num_choices * sample_num, 1, seq_outputs.size(-1))
        c = self.c.expand(batch_size, num_choices * sample_num, 1, -1)

        part_decoder_input_embeds = torch.scatter(part_decoder_input_embeds, dim=2,
                                                  index=part_c_index,
                                                  src=c)

        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size * num_choices * sample_num, part_num + 1, -1)

        pos_num = pos_mask.size(2)
        pos_hidden = pos_hidden.unsqueeze(2).expand(-1, -1, sample_num, -1, -1).reshape(batch_size * num_choices * sample_num,
                                                                                        pos_num, self.t5_config.d_model)
        pos_mask = pos_mask.unsqueeze(2).expand(-1, -1, sample_num, -1).reshape(batch_size * num_choices * sample_num, pos_num)

        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=part_decoder_input_embeds,
            encoder_hidden_states=pos_hidden,
            encoder_attention_mask=pos_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden = torch.gather(decoder_outputs[0].reshape(batch_size, num_choices * sample_num, part_num + 1, self.t5_config.d_model),
                              dim=2, index=part_c_index).reshape(batch_size, num_choices, sample_num, self.t5_config.d_model)

        w_q = self.weight_q(seq_outputs[:, :, 0])
        w_k = self.weight_k(hidden)
        sim = torch.einsum("bnd,bnsd->bns", w_q, w_k).softmax(dim=-1)
        hidden = self.weight_o(torch.einsum("bns,bnsd->bnd", sim, hidden))

        pooled_output = self.dropout(hidden)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def parse_span_rep(seq_outputs: Tensor, index: Tensor, token_mask: Tensor, occur_mask: Tensor):

        # print(f"Index: {index.size()}")
        # print(f"Token mask: {token_mask.size()}")
        # print(f"Occur mask: {occur_mask.size()}")
        # print(f"Seq outputs: {seq_outputs.size()}")

        batch_size, option_num, max_span_num, max_span_occur_num, max_span_len = index.size()
        assert batch_size == seq_outputs.size(0)
        assert option_num == seq_outputs.size(1)
        h = seq_outputs.size(-1)

        flat_index = index.reshape(batch_size, option_num, -1, 1)
        flat_rep = torch.gather(seq_outputs, dim=2, index=flat_index.expand(-1, -1, -1, h))
        flat_rep = flat_rep.reshape(batch_size, option_num, max_span_num, max_span_occur_num, max_span_len, h)

        # print(f"Flat rep: {flat_rep.size()}")

        # Sub-work pooling
        true_token_num = token_mask.sum(dim=4, keepdim=True).to(flat_rep.dtype)
        true_token_num[true_token_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=4) / true_token_num  # (batch_size, option_num, max_span_num, max_span_occur_num, h)

        # Occurrence pooling
        true_occur_num = occur_mask.sum(dim=3, keepdim=True).to(flat_rep.dtype)
        true_occur_num[true_occur_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=3) / true_occur_num  # (batch_size, option_num, max_span_num, h)

        return flat_rep


class RobertaForMultipleChoicePathV2(RobertaForMultipleChoice, ABC):
    def __init__(self, config: RobertaConfig, num_decoder_layers: int, num_extra_encoder_layers: int):
        super().__init__(config)

        self.t5_config = T5Config()
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.num_layers = num_decoder_layers

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.decoder = T5Stack(self.t5_config)
        self.decoder.post_init()

        self.dec_proj = nn.Linear(self.t5_config.d_model, config.hidden_size)

        ex_enc_config = copy.deepcopy(config)
        ex_enc_config.num_hidden_layers = num_extra_encoder_layers
        self.ex_enc_config = ex_enc_config
        self.ex_enc = RobertaEncoder(self.ex_enc_config)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.model_parallel = False
        self.device_map = None

        self.init_weights()

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                op_mask: Tensor = None,
                labels: Tensor = None,
                part_index: Tensor = None,
                part_token_mask: Tensor = None,
                part_occur_mask: Tensor = None,
                part_mask: Tensor = None,
                pos_index: Tensor = None,
                pos_token_mask: Tensor = None,
                pos_occur_mask: Tensor = None,
                pos_mask: Tensor = None,
                part_decoder_input_ids: Tensor = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                use_cache=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.t5_config.use_cache
        num_choices = input_ids.shape[1]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_outputs = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1))

        # (batch_size, option_num, max_span_num, h)
        part_hidden = self.parse_span_rep(seq_outputs, part_index, part_token_mask, part_occur_mask)
        pos_hidden = self.parse_span_rep(seq_outputs, pos_index, pos_token_mask, pos_occur_mask)

        sample_num = part_decoder_input_ids.size(2)
        part_num = part_mask.size(2)
        part_decoder_input_ids = part_decoder_input_ids.reshape(batch_size, num_choices, sample_num * part_num)

        part_decoder_input_embeds = torch.gather(part_hidden, dim=2,
                                                 index=part_decoder_input_ids.unsqueeze(-1).expand(-1, -1, -1, seq_outputs.size(-1)))
        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size * num_choices * sample_num, part_num, -1)

        pos_num = pos_mask.size(2)
        pos_hidden = pos_hidden.unsqueeze(2).expand(-1, -1, sample_num, -1, -1).reshape(batch_size * num_choices * sample_num,
                                                                                        pos_num, self.t5_config.d_model)
        pos_mask = pos_mask.unsqueeze(2).expand(-1, -1, sample_num, -1).reshape(batch_size * num_choices * sample_num, pos_num)

        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=part_decoder_input_embeds,
            encoder_hidden_states=pos_hidden,
            encoder_attention_mask=pos_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        part_assign_hidden = self.dec_proj(decoder_outputs[0].reshape(batch_size * num_choices, sample_num * part_num, -1))

        hidden_states = torch.cat([outputs[0], part_assign_hidden], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_ones(batch_size * num_choices, sample_num * part_num)
        ], dim=1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1],
                                                                                 hidden_states.device)
        head_mask = self.get_head_mask(None, self.ex_enc_config.num_hidden_layers)

        ex_encoder_outputs = self.ex_enc(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = self.dropout(ex_encoder_outputs[0][:, 0])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def parse_span_rep(seq_outputs: Tensor, index: Tensor, token_mask: Tensor, occur_mask: Tensor):

        # print(f"Index: {index.size()}")
        # print(f"Token mask: {token_mask.size()}")
        # print(f"Occur mask: {occur_mask.size()}")
        # print(f"Seq outputs: {seq_outputs.size()}")

        batch_size, option_num, max_span_num, max_span_occur_num, max_span_len = index.size()
        assert batch_size == seq_outputs.size(0)
        assert option_num == seq_outputs.size(1)
        h = seq_outputs.size(-1)

        flat_index = index.reshape(batch_size, option_num, -1, 1)
        flat_rep = torch.gather(seq_outputs, dim=2, index=flat_index.expand(-1, -1, -1, h))
        flat_rep = flat_rep.reshape(batch_size, option_num, max_span_num, max_span_occur_num, max_span_len, h)

        # print(f"Flat rep: {flat_rep.size()}")

        # Sub-work pooling
        true_token_num = token_mask.sum(dim=4, keepdim=True).to(flat_rep.dtype)
        true_token_num[true_token_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=4) / true_token_num  # (batch_size, option_num, max_span_num, max_span_occur_num, h)

        # Occurrence pooling
        true_occur_num = occur_mask.sum(dim=3, keepdim=True).to(flat_rep.dtype)
        true_occur_num[true_occur_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=3) / true_occur_num  # (batch_size, option_num, max_span_num, h)

        return flat_rep


class RobertaForMultipleChoicePathV3(RobertaForMultipleChoice, ABC):
    def __init__(self, config: RobertaConfig, num_decoder_layers: int, num_extra_encoder_layers: int):
        super().__init__(config)

        self.t5_config = T5Config()
        self.t5_config.is_decoder = True
        self.t5_config.is_encoder_decoder = False
        self.t5_config.num_layers = num_decoder_layers

        self.enc_proj = nn.Linear(config.hidden_size, self.t5_config.d_model)

        self.decoder = T5Stack(self.t5_config)
        self.decoder.post_init()

        self.dec_proj = nn.Linear(self.t5_config.d_model, config.hidden_size)

        ex_enc_config = copy.deepcopy(config)
        ex_enc_config.num_hidden_layers = num_extra_encoder_layers
        self.ex_enc_config = ex_enc_config
        self.ex_enc = RobertaEncoder(self.ex_enc_config)

        self.linear1 = nn.Linear(config.hidden_size, 1)

        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )

        self.model_parallel = False
        self.device_map = None

        self.init_weights()

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor = None,
                token_type_ids: Tensor = None,
                op_mask: Tensor = None,
                labels: Tensor = None,
                part_index: Tensor = None,
                part_token_mask: Tensor = None,
                part_occur_mask: Tensor = None,
                part_mask: Tensor = None,
                pos_index: Tensor = None,
                pos_token_mask: Tensor = None,
                pos_occur_mask: Tensor = None,
                pos_mask: Tensor = None,
                part_decoder_input_ids: Tensor = None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                use_cache=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.t5_config.use_cache
        num_choices = input_ids.shape[1]
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(2)

        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        token_type_ids = self.fold_tensor(token_type_ids)

        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        seq_outputs = self.enc_proj(outputs[0].reshape(batch_size, num_choices, seq_len, -1))

        # (batch_size, option_num, max_span_num, h)
        part_hidden = self.parse_span_rep(seq_outputs, part_index, part_token_mask, part_occur_mask)
        pos_hidden = self.parse_span_rep(seq_outputs, pos_index, pos_token_mask, pos_occur_mask)

        sample_num = part_decoder_input_ids.size(2)
        part_num = part_mask.size(2)
        part_decoder_input_ids = part_decoder_input_ids.reshape(batch_size, num_choices, sample_num * part_num)

        part_decoder_input_embeds = torch.gather(part_hidden, dim=2,
                                                 index=part_decoder_input_ids.unsqueeze(-1).expand(-1, -1, -1, seq_outputs.size(-1)))
        part_decoder_input_embeds = part_decoder_input_embeds.reshape(batch_size * num_choices * sample_num, part_num, -1)

        pos_num = pos_mask.size(2)
        pos_hidden = pos_hidden.unsqueeze(2).expand(-1, -1, sample_num, -1, -1).reshape(batch_size * num_choices * sample_num,
                                                                                        pos_num, self.t5_config.d_model)
        pos_mask = pos_mask.unsqueeze(2).expand(-1, -1, sample_num, -1).reshape(batch_size * num_choices * sample_num, pos_num)

        decoder_outputs = self.decoder(
            input_ids=None,
            inputs_embeds=part_decoder_input_embeds,
            encoder_hidden_states=pos_hidden,
            encoder_attention_mask=pos_mask,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        # part_assign_hidden = self.dec_proj(decoder_outputs[0].reshape(batch_size * num_choices, sample_num * part_num, -1))
        part_assign_hidden = self.dec_proj(decoder_outputs[0])
        extended_hidden_states = outputs[0].unsqueeze(1).expand(-1, sample_num, -1, -1).reshape(-1, seq_len, outputs[0].size(-1))
        extended_hidden_states = torch.cat([extended_hidden_states, part_assign_hidden], dim=1)

        # hidden_states = torch.cat([outputs[0], part_assign_hidden], dim=1)
        attention_mask = torch.cat([
            attention_mask.unsqueeze(1).expand(-1, sample_num, -1).reshape(-1, seq_len),
            attention_mask.new_ones(batch_size * num_choices * sample_num, part_num)
        ], dim=1)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, extended_hidden_states.size()[:-1],
                                                                                 extended_hidden_states.device)
        head_mask = self.get_head_mask(None, self.ex_enc_config.num_hidden_layers)

        ex_encoder_outputs = self.ex_enc(
            extended_hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        alpha = torch.softmax(self.linear1(ex_encoder_outputs[0][:, 0]).reshape(batch_size * num_choices, sample_num), dim=-1)
        weighted_hidden = torch.einsum("bs,bsh->bh", alpha, ex_encoder_outputs[0][:, 0])

        pooled_output = self.dropout(weighted_hidden)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices) + (1.0 - op_mask.to(logits.dtype)) * -1e4

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(reshaped_logits, labels)

            if not self.training:
                acc, true_label_num = layers.get_accuracy(reshaped_logits, labels)
                self.eval_metrics.update("acc", val=acc, n=true_label_num)
                self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def parse_span_rep(seq_outputs: Tensor, index: Tensor, token_mask: Tensor, occur_mask: Tensor):

        # print(f"Index: {index.size()}")
        # print(f"Token mask: {token_mask.size()}")
        # print(f"Occur mask: {occur_mask.size()}")
        # print(f"Seq outputs: {seq_outputs.size()}")

        batch_size, option_num, max_span_num, max_span_occur_num, max_span_len = index.size()
        assert batch_size == seq_outputs.size(0)
        assert option_num == seq_outputs.size(1)
        h = seq_outputs.size(-1)

        flat_index = index.reshape(batch_size, option_num, -1, 1)
        flat_rep = torch.gather(seq_outputs, dim=2, index=flat_index.expand(-1, -1, -1, h))
        flat_rep = flat_rep.reshape(batch_size, option_num, max_span_num, max_span_occur_num, max_span_len, h)

        # print(f"Flat rep: {flat_rep.size()}")

        # Sub-work pooling
        true_token_num = token_mask.sum(dim=4, keepdim=True).to(flat_rep.dtype)
        true_token_num[true_token_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=4) / true_token_num  # (batch_size, option_num, max_span_num, max_span_occur_num, h)

        # Occurrence pooling
        true_occur_num = occur_mask.sum(dim=3, keepdim=True).to(flat_rep.dtype)
        true_occur_num[true_occur_num == 0] = 1.
        flat_rep = flat_rep.sum(dim=3) / true_occur_num  # (batch_size, option_num, max_span_num, h)

        return flat_rep




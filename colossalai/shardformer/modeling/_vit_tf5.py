"""ViT forwards for the Transformers 5.17 flat layer/attention layout."""

from typing import List, Optional, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention

from .._utils import get_hf_flash_attention_kwargs, prepare_hf_bidirectional_mask


def ViTModel_pipeline_forward(stage_manager: PipelineStageManager, stage_index: List[int]):
    def pp_forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
        hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        if head_mask is not None:
            raise ValueError("HF 5.17 ViT does not support head_mask")
        if output_attentions or output_hidden_states:
            logging.get_logger(__name__).warning_once("Pipeline ViT does not collect attentions/hidden states.")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")
            pixel_values = pixel_values.to(self.embeddings.patch_embeddings.projection.weight.dtype)
            hidden_states = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
            )
        elif hidden_states is None:
            raise ValueError("hidden_states must be provided to a non-first ViT stage")

        # Explicit masks work for both HF attention and the ColoAttention replacement.
        attention_mask = prepare_hf_bidirectional_mask(self.config, hidden_states, attention_mask)
        indices = stage_index[stage_manager.model_chunk_id] if stage_manager.is_interleave else stage_index
        for i in range(*indices):
            # HF GradientCheckpointingLayer owns checkpointing; do not checkpoint twice.
            hidden_states = self.layers[i](hidden_states, attention_mask=attention_mask, **kwargs)
        if not stage_manager.is_last_stage():
            return {"hidden_states": hidden_states}
        hidden_states = self.layernorm(hidden_states)
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None
        outputs = BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled_output)
        return outputs if return_dict else outputs.to_tuple()

    return pp_forward


def ViTForImageClassification_pipeline_forward(stage_manager: PipelineStageManager, stage_index: List[int]):
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
    from transformers.modeling_outputs import ImageClassifierOutput

    def pp_forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not stage_manager.is_first_stage():
            assert (
                hidden_states is not None
            ), f"Current stage is {stage_manager.stage}, hidden_states should not be None"

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        # not last stage, return hidden_states
        if not stage_manager.is_last_stage():
            return outputs
        else:
            sequence_output = outputs[0]

        # last stage
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return pp_forward


def ViTForMaskedImageModeling_pipeline_forward(stage_manager: PipelineStageManager, stage_index: List[int]):
    import math

    import torch.nn as nn
    from transformers.modeling_outputs import ImageClassifierOutput, MaskedImageModelingOutput

    def pp_forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input."
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        if not stage_manager.is_first_stage():
            assert (
                hidden_states is not None
            ), f"Current stage is {stage_manager.stage}, hidden_states should not be None"

        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        if not stage_manager.is_last_stage():
            return outputs
        else:
            sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return pp_forward


def get_vit_flash_self_attention_forward():
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill(
                ~attention_mask, torch.finfo(query.dtype).min
            )
        output = ColoAttention.attention(
            query,
            key,
            value,
            **get_hf_flash_attention_kwargs(attention_mask),
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output), None

    return forward


def get_jit_fused_vit_layer_forward():
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states, _ = self.attention(self.layernorm_before(hidden_states), attention_mask, **kwargs)
        hidden_states = self.dropout_add(hidden_states, residual, self.dropout.p, self.training)
        residual = hidden_states
        hidden_states = self.mlp(self.layernorm_after(hidden_states))
        return self.dropout_add(hidden_states, residual, self.dropout.p, self.training)

    return forward

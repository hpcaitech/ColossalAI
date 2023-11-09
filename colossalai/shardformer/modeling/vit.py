import math
from typing import List, Optional, Tuple, Union

import torch
from transformers.models.vit.modeling_vit import BaseModelOutput, ViTEncoder
from transformers.utils import logging

from colossalai.pipeline.stage_manager import PipelineStageManager


def _encoder_forward(
    encoder: ViTEncoder,
    start_idx: int,
    end_idx: int,
    hidden_states: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    stage_manager: PipelineStageManager = None,
) -> Union[tuple, BaseModelOutput]:
    for i in range(start_idx, end_idx):
        layer_module = encoder.layer[i]

        layer_head_mask = head_mask[i] if head_mask is not None else None

        if encoder.gradient_checkpointing and encoder.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, False)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                layer_head_mask,
            )
        else:
            layer_outputs = layer_module(hidden_states, layer_head_mask, False)

        hidden_states = layer_outputs[0]
    if not stage_manager.is_last_stage():
        return hidden_states
    else:
        if not return_dict:
            return tuple(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )


def ViTModel_pipeline_forward(stage_manager: PipelineStageManager, stage_index: List[int]):
    from transformers.models.vit.modeling_vit import BaseModelOutputWithPooling

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
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logger = logging.get_logger(__name__)

        # Preprocess passed in arguments
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # TODO(FoolPlayer): maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
            expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
            if pixel_values.dtype != expected_dtype:
                pixel_values = pixel_values.to(expected_dtype)

            embedding_output = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
            )
            hidden_states = embedding_output
        else:
            assert (
                hidden_states is not None
            ), f"Current stage is {stage_manager.stage}, hidden_states should not be None"

        encoder_outputs = _encoder_forward(
            encoder=self.encoder,
            start_idx=stage_index[0],
            end_idx=stage_index[1],
            hidden_states=hidden_states,
            head_mask=head_mask,
            return_dict=return_dict,
            stage_manager=stage_manager,
        )
        if not stage_manager.is_last_stage():
            return {"hidden_states": encoder_outputs}

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    return pp_forward


def ViTForImageClassification_pipeline_forward(stage_manager: PipelineStageManager, stage_index: List[int]):
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
    from transformers.models.vit.modeling_vit import ImageClassifierOutput

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
    from transformers.models.vit.modeling_vit import ImageClassifierOutput, MaskedImageModelingOutput

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
    from transformers.models.vit.modeling_vit import ViTSelfAttention

    from colossalai.kernel.cuda_native import ColoAttention

    def transpose_for_scores(x: torch.Tensor, num_attention_heads, attention_head_size) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(new_x_shape)
        return x

    def forward(
        self: ViTSelfAttention,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = transpose_for_scores(self.key(hidden_states), self.num_attention_heads, self.attention_head_size)
        value_layer = transpose_for_scores(
            self.value(hidden_states), self.num_attention_heads, self.attention_head_size
        )
        query_layer = transpose_for_scores(mixed_query_layer, self.num_attention_heads, self.attention_head_size)

        scale = 1.0 / math.sqrt(self.attention_head_size)
        attention = ColoAttention(
            embed_dim=self.all_head_size, num_heads=self.num_attention_heads, dropout=self.dropout.p, scale=scale
        )
        context_layer = attention(query_layer, key_layer, value_layer)

        outputs = (context_layer,)

        return outputs

    return forward


def get_jit_fused_vit_output_forward():
    from transformers.models.vit.modeling_vit import ViTOutput

    def forward(self: ViTOutput, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout_add(hidden_states, input_tensor, self.dropout.p, self.dropout.training)
        return hidden_states

    return forward

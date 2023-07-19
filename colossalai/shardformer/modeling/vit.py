from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers.models.vit.modeling_vit import BaseModelOutput, BaseModelOutputWithPooling, ViTEncoder

from colossalai.pipeline.stage_manager import PipelineStageManager


def forward_fn(stage_manager: PipelineStageManager, stage_index: List[int]):

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
        if stage_manager.is_first_stage():
            return hidden_states
        else:
            if not return_dict:
                return tuple(hidden_states)
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=None,
                attentions=None,
            )

    def vit_pipeline_forward(
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
            expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
            if pixel_values.dtype != expected_dtype:
                pixel_values = pixel_values.to(expected_dtype)

            embedding_output = self.embeddings(pixel_values,
                                               bool_masked_pos=bool_masked_pos,
                                               interpolate_pos_encoding=interpolate_pos_encoding)
        else:
            assert hidden_states is not None, f"Current stage is {stage_manager.stage}, hidden_states should not be None"

        # Go through encoder
        if stage_manager.is_first_stage():
            hidden_states = _encoder_forward(
                encoder=self.encoder,
                start_idx=stage_index[0],
                end_idx=stage_index[1],
                hidden_states=embedding_output,
                head_mask=head_mask,
                return_dict=return_dict,
                stage_manager=stage_manager,
            )
            return hidden_states
        else:
            encoder_outputs = _encoder_forward(
                encoder=self.encoder,
                start_idx=stage_index[0],
                end_idx=stage_index[1],
                hidden_states=hidden_states,
                head_mask=head_mask,
                return_dict=return_dict,
                stage_manager=stage_manager,
            )

        # Go through rest layers
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

    return vit_pipeline_forward

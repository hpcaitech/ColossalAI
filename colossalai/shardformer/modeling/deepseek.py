from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

# from colossalai.tensor.moe_tensor.moe_info import MoeParallelInfo
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_flash_attn_2_available, logging

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import MoeInGradScaler, MoeOutGradScaler, all_to_all_uneven
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.shard import ShardConfig
from colossalai.shardformer.shard.utils import set_tensors_to_none


# copied from modeling_deepseek.py
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class EPDeepseekMoE(nn.Module):
    def __init__(self):
        super(EPDeepseekMoE, self).__init__()

    def setup_ep(self, ep_group: ProcessGroup):
        ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group) if ep_group is not None else 1
        self.ep_rank = dist.get_rank(ep_group) if ep_group is not None else 0
        self.num_experts = self.config.n_routed_experts
        assert self.num_experts % self.ep_size == 0
        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]
        set_tensors_to_none(self.experts, exclude=set(held_experts))
        for p in self.experts.parameters():
            p.ep_group = ep_group

    @staticmethod
    def from_native_module(module: Union["DeepseekMoE", "DeepseekMLP"], *args, **kwargs) -> "EPDeepseekMoE":
        LazyInitContext.materialize(module)
        if module.__class__.__name__ == "DeepseekMLP":
            return module
        module.__class__ = EPDeepseekMoE
        assert "ep_group" in kwargs, "You should pass ep_group in SubModuleReplacementDescription via shard_config!!"
        module.setup_ep(kwargs["ep_group"])
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        topk_experts_idx, topk_experts_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # [t0, t1, t2 ...]
        hidden_states = hidden_states.repeat_interleave(
            self.num_experts_per_tok, dim=0
        )  # after repeat_interleave: [t0 t0 t1 t1 t2 t2 ... ]

        flat_topk_experts_idx = topk_experts_idx.view(-1)  # [e0 e1 e2 ...]
        # The elements of flat_topk_token_idx are token ids, which are arranged in ascending order of expert ids.
        flat_topk_token_idx = flat_topk_experts_idx.argsort()

        # Now we adjust the order of the hidden states, also in ascending order of expert id
        dispatch_states = hidden_states[flat_topk_token_idx]
        input_split_sizes = flat_topk_experts_idx.bincount(minlength=self.num_experts)  # [n0, n1, n2, n3]
        output_split_sizes = torch.zeros_like(input_split_sizes)

        # [n0, n1, n2, n3] [m0, m1, m2, m3] -> [n0, n1, m0, m1] [n2, n3, m2, m3]
        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)

        input_split_list = input_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_split_list = output_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_states, _ = all_to_all_uneven(dispatch_states, input_split_list, output_split_list, self.ep_group)
        output_states = MoeInGradScaler.apply(output_states, self.ep_size)

        if output_states.size(0) > 0:
            if self.num_experts_per_ep == 1:
                expert = self.experts[self.expert_start_idx]
                output_states = expert(output_states)
            else:
                output_states_splits = output_states.split(output_split_sizes.tolist())
                output_states_list = []
                for i, split_states in enumerate(output_states_splits):
                    if split_states.size(0) == 0:  # no token routed to this experts
                        continue
                    expert = self.experts[self.expert_start_idx + i % self.num_experts_per_ep]
                    split_states = expert(split_states)
                    output_states_list.append(split_states)
                output_states = torch.cat(output_states_list)
        output_states = MoeOutGradScaler.apply(output_states, self.ep_size)
        dispatch_states, _ = all_to_all_uneven(output_states, output_split_list, input_split_list, self.ep_group)
        recover_token_idx = torch.empty_like(flat_topk_token_idx)
        recover_token_idx[flat_topk_token_idx] = torch.arange(
            flat_topk_token_idx.size(0), device=flat_topk_token_idx.device
        )

        output_hidden_states = dispatch_states[recover_token_idx]  # t0 t0 t1 t1 t2 t2
        output_hidden_states = output_hidden_states.view(-1, self.num_experts_per_tok, orig_shape[-1])
        output_hidden_states = (output_hidden_states * topk_experts_weight[:, :, None]).sum(dim=-2)  # (B*S, h)
        output_hidden_states = output_hidden_states.view(*orig_shape)
        output_hidden_states = AddAuxiliaryLoss.apply(output_hidden_states, aux_loss)
        if self.config.n_shared_experts is not None:
            output_hidden_states = output_hidden_states + self.shared_experts(identity)
        return output_hidden_states


class DeepseekPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Llama models
    under pipeline setting.
    """

    @staticmethod
    def deepseek_model_forward(
        self: "DeepseekModel",
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if stage_manager.is_first_stage():
            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device

        seq_length_with_past = seq_length
        past_key_values_length = 0

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions, for the first stage, hidden_states is the input embeddings,
        # for the other stages, hidden_states is the output of the previous stage
        if is_flash_attn_2_available():
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx], start=start_idx):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        if stage_manager.is_last_stage():
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # always return dict for imediate stage
        return {
            "hidden_states": hidden_states,
        }

    @staticmethod
    def deepseek_for_causal_lm_forward(
        self: "DeepseekForCausalLM",
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MixtralForCausalLM

        >>> model = DeepseekForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        logger = logging.get_logger(__name__)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = DeepseekPipelineForwards.deepseek_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
        )
        past_key_values = None

        if stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=outputs[0],
                attentions=None,
            )
        else:
            out = {}
            hidden_states = outputs.get("hidden_states")
            out["hidden_states"] = hidden_states
            return out

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import (
    DPGradScalerIn,
    DPGradScalerOut,
    EPGradScalerIn,
    EPGradScalerOut,
    all_to_all_uneven,
)
from colossalai.shardformer.layer.linear import ParallelModule
from colossalai.shardformer.shard.utils import set_tensors_to_none
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_group


class EpDeepseekV3MoE(ParallelModule):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    def setup_process_groups(
        self,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
    ):
        assert moe_dp_group is not None
        assert ep_group is not None

        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = self.config.n_routed_experts
        assert self.num_experts % self.ep_size == 0

        self.ep_group = ep_group
        self.num_experts_per_ep = self.num_experts // self.ep_size
        self.experts_per_rank = self.num_experts_per_ep
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep
        held_experts = self.experts[self.expert_start_idx : self.expert_start_idx + self.num_experts_per_ep]

        set_tensors_to_none(self.experts, exclude=set(held_experts))

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = dist.get_world_size(moe_dp_group)

        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)

    @staticmethod
    def from_native_module(
        module,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        *args,
        **kwargs,
    ) -> "EpDeepseekV3MoE":
        if module.__class__.__name__ != "DeepseekV3MLP":
            module.__class__ = EpDeepseekV3MoE
            module.setup_process_groups(moe_dp_group, ep_group)
        LazyInitContext.materialize(module)
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_forward(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.ep_group)

            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(1).tolist()
            input_split_sizes = tokens_per_ep_rank.tolist()

            gathered_tokens, _ = all_to_all_uneven(sorted_tokens, input_split_sizes, output_splits, self.ep_group)
            tokens_per_expert_post_gather = tokens_per_expert_group.view(self.ep_size, self.experts_per_rank).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather

            # moe-dp related code
            activate_experts = tokens_per_expert_post_gather > 0
            activate_experts = activate_experts.int()
            dist.all_reduce(activate_experts, group=self.moe_dp_group)

            # ep related code
            sorted_tokens = EPGradScalerIn.apply(sorted_tokens, self.ep_size)

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            # moe-dp related code
            tokens_for_this_expert = DPGradScalerIn.apply(tokens_for_this_expert, self.moe_dp_size, activate_experts[i])
            expert_out = expert(tokens_for_this_expert)
            # moe-dp related code
            expert_out = DPGradScalerOut.apply(expert_out, self.moe_dp_size, activate_experts[i])
            outputs.append(expert_out)
            start_idx = end_idx

        if len(outputs) > 0:
            outs = torch.cat(outputs, dim=0)
        else:
            assert sorted_tokens.numel() == 0, f"sorted_tokens: should be empty, but got {sorted_tokens.shape}"
            outs = sorted_tokens

        if self.ep_size > 1:
            outs = EPGradScalerOut.apply(outs, self.ep_size)
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens, _ = all_to_all_uneven(new_x, output_splits, input_split_sizes, self.ep_group)
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return final_out


def deepseek_v3_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and i > 0:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

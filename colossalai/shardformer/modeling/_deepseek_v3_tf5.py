from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.output_capturing import capture_outputs

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import EPGradScalerIn, EPGradScalerOut, all_to_all_uneven
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer.linear import ParallelModule
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_fused, set_moe_tensor_ep_group


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
        self.ep_group = ep_group

        # v5 stores experts as fused 3D parameters (`gate_up_proj [E,2I,H]` / `down_proj [E,H,I]`), so
        # `num_experts` lives on `self.experts`, not on the block (docs/30 §二).
        num_experts = self.experts.num_experts
        if num_experts % self.ep_size != 0:
            raise ValueError("The number of experts must be divisible by the number of expert parallel groups.")

        self.num_experts = num_experts
        self.num_experts_per_ep = num_experts // self.ep_size
        self.expert_start_idx = self.ep_rank * self.num_experts_per_ep

        # primitive ① (docs/30 §4.1): slice the fused params to the local experts and release the rest
        # (P5). `.clone()` is required, not `.contiguous()`: the dim-0 slice of a contiguous tensor is
        # already contiguous, so `.contiguous()` would return the view and keep the full `[E, ...]` alive.
        experts = self.experts
        s = self.expert_start_idx
        n = self.num_experts_per_ep
        # Swap the storage but keep the Parameter objects: optimizers are built from the unsharded model and
        # the MoE plugin only prunes (never re-adds) optimizer params, so a new Parameter would drop out of training.
        experts.gate_up_proj.data = experts.gate_up_proj.data[s : s + n].clone()
        experts.down_proj.data = experts.down_proj.data[s : s + n].clone()
        experts.num_experts = n

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = dist.get_world_size(moe_dp_group)

        # primitive ② (docs/30 §4.2): mark the sliced fused params so the sharded loader slices dim 0
        # (the expert dimension) by `ep_group`.
        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)
            set_moe_tensor_ep_fused(p)

    @staticmethod
    def from_native_module(
        module,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        *args,
        **kwargs,
    ) -> "EpDeepseekV3MoE":
        # Materialize first, then slice: primitive ① clones the fused params, which must already be
        # real tensors under lazy-init (mirrors EPMixtralSparseMoeBlock, docs/30 §四).
        LazyInitContext.materialize(module)
        # The first `first_k_dense_replace` layers keep a dense `DeepseekV3MLP` (no experts); only
        # the MoE layers (`DeepseekV3MoE`) carry the fused 3D params to slice.
        if module.__class__.__name__ != "DeepseekV3MLP":
            module.__class__ = EpDeepseekV3MoE
            module.setup_process_groups(moe_dp_group, ep_group)
        return module

    def _expert_forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        r"""Run one local expert on its dispatched tokens (docs/30 §4.3).

        Mirrors v5's ``DeepseekV3Experts.forward``: ``gate, up = linear(x, gate_up_proj[e]).chunk(2,
        dim=-1)``, ``act_fn(gate) * up``, then ``linear(down_proj[e])``. The fused params are already
        sliced to ``[E/ep, ...]``, so ``expert_idx`` is a local index.
        """
        gate, up = F.linear(x, self.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        return F.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[expert_idx])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # v5's `DeepseekV3TopkRouter` returns the (logits, top-k weights, top-k indices) triple; group
        # scoring / top-k / normalisation / `routed_scaling_factor` are already applied inside it, so
        # there is no local routing to redo here.
        _, topk_weights, topk_indices = self.gate(hidden_states)
        routing_weights = topk_weights.to(hidden_states.dtype)

        # Unlike Mixtral, `DeepseekV3MoE` carries no `self.top_k`; it lives on the router.
        top_k = self.gate.top_k

        selected_experts = topk_indices.t().reshape(-1)
        selected_experts_idx = selected_experts.argsort()
        dispatch_states = hidden_states.repeat(top_k, 1)[selected_experts_idx]
        input_split_sizes = selected_experts.bincount(minlength=self.num_experts)

        output_split_sizes = torch.zeros_like(input_split_sizes)

        dist.all_to_all_single(output_split_sizes, input_split_sizes, group=self.ep_group)

        input_split_list = input_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()
        output_split_list = output_split_sizes.view(self.ep_size, self.num_experts_per_ep).sum(dim=-1).tolist()

        output_states, _ = all_to_all_uneven(
            dispatch_states,
            input_split_list,
            output_split_list,
            self.ep_group,
        )
        # compute expert output
        output_states = EPGradScalerIn.apply(output_states, self.ep_size)
        if output_states.size(0) > 0:
            if self.num_experts_per_ep == 1:
                # no need to split
                output_states = self._expert_forward(output_states, 0)
            else:
                output_states_splits = output_states.split(output_split_sizes.tolist())
                output_states_list = []
                for i, split_states in enumerate(output_states_splits):
                    if split_states.size(0) == 0:
                        continue
                    expert_idx = i % self.num_experts_per_ep
                    split_states = self._expert_forward(split_states, expert_idx)
                    output_states_list.append(split_states)
                output_states = torch.cat(output_states_list)

        output_states = EPGradScalerOut.apply(output_states, self.ep_size)
        dispatch_states, _ = all_to_all_uneven(output_states, output_split_list, input_split_list, self.ep_group)

        recover_experts_idx = torch.empty_like(selected_experts_idx)
        recover_experts_idx[selected_experts_idx] = torch.arange(
            selected_experts_idx.size(0), device=selected_experts_idx.device
        )
        dispatch_states = dispatch_states[recover_experts_idx]
        k_hidden_states = dispatch_states.chunk(top_k)
        output_states = k_hidden_states[0] * routing_weights[:, 0, None]
        for i in range(1, top_k):
            output_states += k_hidden_states[i] * routing_weights[:, i, None]
        output_states = output_states.view(*orig_shape)

        if self.config.n_shared_experts is not None:
            output_states = output_states + self.shared_experts(identity)
        return output_states


# v5 records `attentions` / `hidden_states` declaratively and injects them via `@capture_outputs`.
# v5's own forward carries that decorator and ours replaces it, so it has to be put back. Unlike
# Mixtral, the recorded classes survive here (the policy keeps `self_attn`).
@capture_outputs
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
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
    hidden_states_internal: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # Not consumed here under v5: the `@capture_outputs` wrapper reads these from `kwargs` to
    # decide what to record, then injects the result. Kept for API shape, and deliberately not
    # resolved -- a resolved value would look like this body accumulates them itself.
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
        # v5 dropped `DynamicCache.from_legacy_cache` / `to_legacy_cache` and
        # `Cache.get_usable_length`; `ddp_cache_data` still takes the per-layer
        # `(key, value[, sliding_window])` tuples, and the length comes from `get_seq_length()`.
        if past_key_values is None:
            past_key_values = DynamicCache()
        elif not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache(past_key_values)
        past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if stage_manager is None or stage_manager.is_first_stage():
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
    else:
        inputs_embeds = hidden_states_internal

    # v5 dropped `DeepseekV3Model._update_causal_mask` in favour of `create_causal_mask`, with
    # `allow_is_causal_skip` left at its default: DeepSeek-v3 runs v5's native attention (the
    # policy keeps `self_attn`), which lets the backend decide, and `_use_flash_attention_2` --
    # the attribute the old manual dispatch read -- no longer exists.
    attention_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    # Neither is accumulated by hand under v5: `DecoderLayer.forward` returns a bare tensor and
    # `@capture_outputs` overwrites both fields afterwards. Accumulating would also give wrong
    # results for v5's new "layer-index list" form of `output_hidden_states`.
    all_hidden_states = None
    all_self_attns = None
    next_decoder_cache = None

    if stage_index is not None:
        start_idx, end_idx = stage_index
    else:
        start_idx, end_idx = 0, len(self.layers)
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
    for i, decoder_layer in enumerate(self.layers[start_idx:end_idx], start=start_idx):
        if self.gradient_checkpointing:
            # v5 moves the recompute decision into `GradientCheckpointingLayer.__call__`, driven by
            # a per-layer flag (which is what `gradient_checkpointing_enable()` sets -- the flag
            # relies on it). Calling `_gradient_checkpointing_func` directly, as before, now
            # misaligns the trailing positional args and fails inside attention. Layer 0 keeps
            # its old exemption.
            decoder_layer.gradient_checkpointing = i > 0

        # Keyword names must match v5's `DeepseekV3DecoderLayer.forward`: the old `past_key_value=`
        # (singular) and `output_attentions=` no longer exist and are swallowed by `**kwargs` --
        # that is how the KV cache was being lost.
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        # v5's `DeepseekV3DecoderLayer.forward` returns a bare tensor (v4 returned a tuple), so
        # the old `layer_outputs[0]` would index into the batch dim instead.
        hidden_states = layer_outputs

        if use_cache:
            # v5 updates the passed-in `Cache` in place instead of returning it
            next_decoder_cache = past_key_values

    if stage_manager is None or stage_manager.is_last_stage():
        hidden_states = self.norm(hidden_states)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache
    if stage_manager is not None and not stage_manager.is_last_stage():
        return {
            "hidden_states_internal": hidden_states,
        }
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def deepseek_v3_for_causal_lm_forward(
    self,
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
    logits_to_keep: Union[int, torch.Tensor] = 0,
    stage_manager: Optional[PipelineStageManager] = None,
    stage_index: Optional[List[int]] = None,
    hidden_states_internal: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.
    Returns:
    Example:
    ```python
    >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM
    >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")
    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    # v5 moves `output_attentions` / `output_hidden_states` into `**kwargs`, so pull them back out
    # here -- otherwise the same key travels both ways and collides on the way down.
    output_attentions = kwargs.pop("output_attentions", output_attentions)
    output_hidden_states = kwargs.pop("output_hidden_states", output_hidden_states)
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # These must be resolved booleans, never `None`: the downstream `@capture_outputs` wrapper
    # tests `kwargs.get("output_attentions", config.output_attentions)`, and an explicit `None`
    # does not fall back to the config -- it silently turns recording off.
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    # `return_dict` is deliberately not forwarded: the wrapper pops it and unconditionally calls
    # `outputs.to_tuple()` when it is False, but the non-last PP stage returns a plain dict, which
    # has no `to_tuple`. The parameter below is therefore unreachable from outside, kept for API.
    outputs = deepseek_v3_model_forward(
        self.model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        stage_manager=stage_manager,
        stage_index=stage_index,
        hidden_states_internal=hidden_states_internal,
    )
    if stage_manager is not None and not stage_manager.is_last_stage():
        return outputs

    hidden_states = outputs[0]

    # v5 addition: logits for the last `logits_to_keep` positions only (`0` = all, the old behaviour).
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
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
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

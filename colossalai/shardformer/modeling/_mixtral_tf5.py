import inspect
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.mixtral.modeling_mixtral import (
    MixtralModel,
    MixtralSparseMoeBlock,
    MixtralTopKRouter,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    apply_rotary_pos_emb,
    load_balancing_loss_func,
)
from transformers.utils import is_flash_attn_2_available, logging
from transformers.utils.output_capturing import capture_outputs

from colossalai.lazy import LazyInitContext
from colossalai.moe._operation import EPGradScalerIn, EPGradScalerOut, all_to_all_uneven
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_forward_split_backward,
    linear_with_async_comm,
    reduce_forward,
    split_forward_gather_backward,
)
from colossalai.shardformer.layer.linear import Linear1D_Col, LinearWithGradAccum, ParallelModule
from colossalai.shardformer.shard import ShardConfig
from colossalai.tensor.d_tensor import init_tensor_as_customization_distributed
from colossalai.tensor.moe_tensor.api import set_moe_tensor_ep_fused, set_moe_tensor_ep_group

# v4-era flag, read by `get_mixtral_flash_attention_forward` for a one-off warning. It is only
# populated below when flash-attn is actually installed; default to "supported" so the reference
# stays in scope (and no spurious sliding-window warning fires) on flash-attn-free environments.
_flash_supports_window_size = True

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


def _slice_fused_expert_param(
    tensor: torch.Tensor,
    name: str,
    expert_start_idx: int,
    num_experts_per_ep: int,
    num_experts: int,
    tp_rank: int,
    tp_size: int,
    local_shape: torch.Size,
) -> torch.Tensor:
    """Select this rank's EP experts and TP columns from a full or EP-local checkpoint tensor."""
    if tensor.shape == local_shape:
        return tensor
    if tensor.ndim != len(local_shape):
        raise ValueError(f"Unexpected {name} rank in the Mixtral expert checkpoint.")

    if tensor.shape[0] == num_experts:
        tensor = tensor[expert_start_idx : expert_start_idx + num_experts_per_ep]
    elif tensor.shape[0] != num_experts_per_ep:
        raise ValueError(f"Unexpected expert dimension in the Mixtral {name} checkpoint.")

    if tp_size > 1:
        if name == "gate_up_proj":
            if tensor.shape[1] % 2 != 0:
                raise ValueError("The fused gate_up_proj dimension must contain an even gate/up pair.")
            intermediate_size = tensor.shape[1] // 2
            if intermediate_size % tp_size != 0:
                raise ValueError("The fused gate_up_proj dimension must be divisible by tp_size.")
            tp_partition_size = intermediate_size // tp_size
            tp_start = tp_rank * tp_partition_size
            tensor = torch.cat(
                (
                    tensor.narrow(1, tp_start, tp_partition_size),
                    tensor.narrow(1, intermediate_size + tp_start, tp_partition_size),
                ),
                dim=1,
            )
        else:
            if tensor.shape[2] % tp_size != 0:
                raise ValueError("The fused down_proj intermediate dimension must be divisible by tp_size.")
            tp_partition_size = tensor.shape[2] // tp_size
            tensor = tensor.narrow(2, tp_rank * tp_partition_size, tp_partition_size)

    if tensor.shape != local_shape:
        raise ValueError(f"The sliced Mixtral {name} shape does not match the local expert parameter.")
    return tensor.contiguous()


def _shard_param_in_place(param: torch.nn.Parameter, local: torch.Tensor) -> None:
    """Swap a parameter's storage for its local shard while keeping the Parameter object.

    Optimizers are built from the unsharded model and HybridParallel/MoE plugins only prune (never
    re-add) optimizer params, so replacing the Parameter object would silently drop it from training.
    """
    param.data = local


def _register_fused_expert_checkpoint_hook(
    experts: torch.nn.Module,
    expert_start_idx: int,
    num_experts_per_ep: int,
    num_experts: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Shard full fused-expert parameters before they reach the local EP module."""

    def _slice_full_expert_params(
        module: torch.nn.Module,
        state_dict,
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        for name in ("gate_up_proj", "down_proj"):
            key = prefix + name
            checkpoint_param = state_dict.get(key)
            local_param = getattr(module, name)
            if checkpoint_param is None:
                continue
            if checkpoint_param.shape == local_param.shape:
                continue
            try:
                state_dict[key] = _slice_fused_expert_param(
                    checkpoint_param,
                    name,
                    expert_start_idx,
                    num_experts_per_ep,
                    num_experts,
                    tp_rank,
                    tp_size,
                    local_param.shape,
                )
            except ValueError:
                continue

    experts.register_load_state_dict_pre_hook(_slice_full_expert_params)


def _mark_fused_expert_tp_shard(
    param: torch.nn.Parameter,
    name: str,
    tp_group: ProcessGroup,
    expert_start_idx: int,
    num_experts_per_ep: int,
    num_experts: int,
) -> None:
    """Attach checkpoint gather/scatter functions to a manually TP-sharded fused expert parameter."""
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    local_shape = param.shape

    def shard_fn(tensor: torch.Tensor) -> torch.Tensor:
        return _slice_fused_expert_param(
            tensor,
            name,
            expert_start_idx,
            num_experts_per_ep,
            num_experts,
            tp_rank,
            tp_size,
            local_shape,
        )

    def gather_fn(tensor: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=tp_group)
        if name == "gate_up_proj":
            partition_size = tensor.shape[1] // 2
            gate = torch.cat([shard[:, :partition_size] for shard in gathered], dim=1)
            up = torch.cat([shard[:, partition_size:] for shard in gathered], dim=1)
            return torch.cat((gate, up), dim=1)
        return torch.cat(gathered, dim=2)

    init_tensor_as_customization_distributed(param, shard_fn, gather_fn)


class _MixtralTopKRouterMixin:
    r"""Sharded / grad-accumulation variants of the v5 ``MixtralTopKRouter``.

    Only the matmul is replaced; the post-processing matches v5's
    ``MixtralTopKRouter.forward`` line for line.

    Args:
        hidden_dim (int): second dimension of the router weight, used to reshape the input
            into ``(N, hidden_dim)``.
        top_k (int): number of experts selected per token.

    v5 collects `router_logits` declaratively: `MixtralModel._can_record_outputs` declares
    `OutputRecorder(MixtralTopKRouter, index=0)` and the capturing hook is matched with
    `isinstance` (`output_capturing.py:165`). Both subclasses below therefore also inherit from
    `MixtralTopKRouter`, so the hook attaches to the sharded router as well and records the first
    element of its (logits, scores, indices) triple. No extra communication is needed for that:
    `Linear1D_Col` runs with `gather_output=True`, so the expert dimension of the logits is already
    gathered back on every rank.
    """

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = super().forward(hidden_states)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value, router_indices


# `MixtralTopKRouter` is the last base on purpose: it is only there for the `isinstance` match of
# v5's capturing hook. The mixin must come first so that `super().forward()` inside it reaches
# `Linear1D_Col.forward` (sharded matmul + gather); putting the native router before the parallel
# linear would silently take v5's `F.linear` path instead, i.e. topk over a single rank's experts.
class MixtralTopKRouter1D(_MixtralTopKRouterMixin, Linear1D_Col, MixtralTopKRouter):
    r"""Tensor-parallel Mixtral router.

    v5 replaced ``MixtralSparseMoeBlock.gate`` with ``MixtralTopKRouter`` and unpacks a triple
    from it::

        _, top_k_weights, top_k_index = self.gate(hidden_states)

    ``Linear1D_Col`` returns a single tensor, so substituting it directly would silently compute
    the wrong thing: unpacking an ``[N, E]`` output iterates over dim 0 (the tokens), which raises
    only when ``N == 3``. This class keeps the weight sharded over the expert dimension while
    matching v5's forward contract.

    The weight stays under the name ``weight`` (the sharding writes back in place), so the state
    dict key ``...mlp.gate.weight`` matches v5's HF key.

    Inherits from `MixtralTopKRouter` so that v5's output-capturing hook recognises it; see the
    mixin docstring for why that is the only requirement (the logits are already gathered).
    """

    @staticmethod
    def from_native_module(
        module: "MixtralTopKRouter", process_group: ProcessGroup = None, **kwargs
    ) -> "MixtralTopKRouter1D":
        r"""Convert a native ``MixtralTopKRouter`` to a tensor-parallel one.

        Args:
            module (MixtralTopKRouter): the native router to be converted.
            process_group (ProcessGroup): the process group of tensor parallelism.
            **kwargs: passed to ``Linear1D_Col`` (e.g. ``fp8_communication`` / ``use_zbv``).
        """
        LazyInitContext.materialize(module)

        top_k, hidden_dim = module.top_k, module.hidden_dim
        num_experts, in_features = module.weight.shape

        # Under sequence parallelism (split_gather / ring / all_to_all) the sequence — not the batch
        # — is split across the TP ranks, so the standard expert-dim gather below would pair each
        # local token with a *different* rank's local token. The router weight is tiny
        # ([num_experts, hidden]), so keep it unsharded: the native forward already computes the full
        # logits for the local tokens, and no gather is needed.
        if kwargs.get("seq_parallel_mode", None) is not None:
            return module

        tp_size = dist.get_world_size(process_group) if process_group is not None else 1
        if num_experts < tp_size:
            # Same as `Linear1D_Col.from_native_module` (linear.py:301-303): skip sharding when it
            # would not divide evenly.
            return module

        # `Linear1D_Col` shards by writing back in place (`sharded_tensor_to_existing_param`), so
        # the router's own `Parameter` ends up holding the shard.
        shim = torch.nn.Linear(in_features, num_experts, bias=False)
        shim.weight = module.weight

        kwargs = dict(kwargs)
        kwargs.setdefault("gather_output", True)
        router = Linear1D_Col.from_native_module(shim, process_group, **kwargs)
        router.__class__ = MixtralTopKRouter1D
        router.top_k = top_k
        router.hidden_dim = hidden_dim
        return router


class MixtralTopKRouterWithGradAccum(_MixtralTopKRouterMixin, LinearWithGradAccum, MixtralTopKRouter):
    r"""Mixtral router for Zero-Bubble-V (``use_zbv``, no TP).

    Same reasoning as ``MixtralTopKRouter1D``: the ZBV branch used to substitute
    ``LinearWithGradAccum`` for the gate, which breaks v5's triple contract in the same way. Only
    the matmul is swapped, and nothing is sharded, so the logits keep v5's shapes.

    Inherits from `MixtralTopKRouter` so that v5's output-capturing hook recognises it; see the
    mixin docstring for why that is the only requirement (nothing is sharded here, so the logits
    are complete by construction).
    """

    @staticmethod
    def from_native_module(module: "MixtralTopKRouter", **kwargs) -> "MixtralTopKRouterWithGradAccum":
        r"""Convert a native ``MixtralTopKRouter`` to a grad-accumulation one."""
        LazyInitContext.materialize(module)

        top_k, hidden_dim = module.top_k, module.hidden_dim
        num_experts, in_features = module.weight.shape

        shim = torch.nn.Linear(in_features, num_experts, bias=False)
        shim.weight = module.weight

        router = LinearWithGradAccum.from_native_module(shim, **kwargs)
        router.__class__ = MixtralTopKRouterWithGradAccum
        router.top_k = top_k
        router.hidden_dim = hidden_dim
        return router


class EPMixtralSparseMoeBlock(ParallelModule):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    def setup_process_groups(
        self,
        tp_group: ProcessGroup,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        fp8_communication: bool = False,
        use_zbv: bool = False,
    ):
        assert tp_group is not None
        assert moe_dp_group is not None
        assert ep_group is not None

        # setup ep group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_group = ep_group
        self.fp8_communication = fp8_communication
        self.use_zbv = use_zbv

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
        _shard_param_in_place(experts.gate_up_proj, experts.gate_up_proj.data[s : s + n].clone())
        _shard_param_in_place(experts.down_proj, experts.down_proj.data[s : s + n].clone())
        experts.num_experts = n

        # setup moe_dp group
        self.moe_dp_group = moe_dp_group
        self.moe_dp_size = moe_dp_group.size()

        # setup global tp group
        self.tp_group = tp_group
        self.tp_size = tp_group.size()
        self.tp_rank = dist.get_rank(tp_group)
        if self.tp_size > 1:
            gate_up_proj = experts.gate_up_proj.data
            down_proj = experts.down_proj.data
            if gate_up_proj.shape[1] % 2 != 0:
                raise ValueError("The fused gate_up_proj dimension must contain an even gate/up pair.")
            intermediate_size = gate_up_proj.shape[1] // 2
            if intermediate_size != down_proj.shape[2]:
                raise ValueError("The fused gate_up_proj and down_proj intermediate dimensions must match.")
            if intermediate_size % self.tp_size != 0:
                raise ValueError("The fused intermediate dimension must be divisible by tp_size.")
            if down_proj.shape[2] % self.tp_size != 0:
                raise ValueError("The fused down_proj intermediate dimension must be divisible by tp_size.")

            gate_up_partition_size = intermediate_size // self.tp_size
            down_partition_size = down_proj.shape[2] // self.tp_size
            tp_start = self.tp_rank * gate_up_partition_size
            _shard_param_in_place(
                experts.gate_up_proj,
                torch.cat(
                    (
                        gate_up_proj[:, tp_start : tp_start + gate_up_partition_size],
                        gate_up_proj[
                            :, intermediate_size + tp_start : intermediate_size + tp_start + gate_up_partition_size
                        ],
                    ),
                    dim=1,
                ).clone(),
            )
            tp_start = self.tp_rank * down_partition_size
            _shard_param_in_place(experts.down_proj, down_proj[:, :, tp_start : tp_start + down_partition_size].clone())

            # These manual slices are not DTensors; checkpoint sharding needs explicit TP gather metadata.
            _mark_fused_expert_tp_shard(experts.gate_up_proj, "gate_up_proj", tp_group, s, n, num_experts)
            _mark_fused_expert_tp_shard(experts.down_proj, "down_proj", tp_group, s, n, num_experts)

        _register_fused_expert_checkpoint_hook(experts, s, n, num_experts, tp_rank=self.tp_rank, tp_size=self.tp_size)

        # primitive ② (docs/30 §4.2): mark the sliced fused params so the sharded loader slices dim 0
        # (the expert dimension) by `ep_group`.
        for p in self.experts.parameters():
            set_moe_tensor_ep_group(p, ep_group)
            set_moe_tensor_ep_fused(p)

    @staticmethod
    def from_native_module(
        module: MixtralSparseMoeBlock,
        tp_group: ProcessGroup,
        moe_dp_group: ProcessGroup,
        ep_group: ProcessGroup,
        *args,
        **kwargs,
    ) -> "EPMixtralSparseMoeBlock":
        # TODO: better init
        LazyInitContext.materialize(module)
        module.__class__ = EPMixtralSparseMoeBlock
        fp8_communication = kwargs.get("fp8_communication", False)
        use_zbv = kwargs.get("use_zbv", False)
        module.setup_process_groups(tp_group, moe_dp_group, ep_group, fp8_communication, use_zbv)
        return module

    def _expert_forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        r"""Run one local expert on its dispatched tokens (docs/30 §4.3).

        Mirrors v5's ``MixtralExperts.forward``: ``gate, up = linear(x, gate_up_proj[e]).chunk(2,
        dim=-1)``, ``act_fn(gate) * up``, then ``linear(down_proj[e])``. The fused params are already
        sliced to ``[E/ep, ...]``, so ``expert_idx`` is a local index. With TP over experts, each local
        ``gate_up_proj`` keeps matching gate/up slices and ``down_proj`` keeps the corresponding input slice;
        the partial output is then reduced.
        """
        gate_up_weight = self.experts.gate_up_proj[expert_idx]
        if self.tp_size == 1:
            gate_up = F.linear(x, gate_up_weight)
        else:
            gate_up = linear_with_async_comm(
                x,
                gate_up_weight,
                None,
                self.tp_group,
                async_grad_allreduce=True,
                fp8_communication=self.fp8_communication,
                use_zbv=self.use_zbv,
            )
        gate, up = gate_up.chunk(2, dim=-1)
        output = F.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[expert_idx])
        if self.tp_size > 1:
            output = reduce_forward(output, self.tp_group, fp8_communication=self.fp8_communication)
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # v5's `MixtralTopKRouter` returns the (logits, normalised top-k weights, top-k indices) triple;
        # softmax / top-k / normalisation are already applied, so there is no local routing to redo here.
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        routing_weights = top_k_weights.to(hidden_states.dtype)

        selected_experts = top_k_index.t().reshape(-1)
        selected_experts_idx = selected_experts.argsort()
        dispatch_states = hidden_states.repeat(self.top_k, 1)[selected_experts_idx]
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
            fp8_communication=self.fp8_communication,
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
        dispatch_states, _ = all_to_all_uneven(
            output_states, output_split_list, input_split_list, self.ep_group, fp8_communication=self.fp8_communication
        )

        recover_experts_idx = torch.empty_like(selected_experts_idx)
        recover_experts_idx[selected_experts_idx] = torch.arange(
            selected_experts_idx.size(0), device=selected_experts_idx.device
        )
        dispatch_states = dispatch_states[recover_experts_idx]
        k_hidden_states = dispatch_states.chunk(self.top_k)
        output_states = k_hidden_states[0] * routing_weights[:, 0, None]
        for i in range(1, self.top_k):
            output_states += k_hidden_states[i] * routing_weights[:, i, None]
        output_states = output_states.reshape(batch_size, sequence_length, hidden_dim)
        return output_states


class MixtralPipelineForwards:
    """
    This class serves as a micro library for forward function substitution of Mixtral models
    under pipeline setting.
    """

    @staticmethod
    def mixtral_model_forward(
        self: MixtralModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        past_router_logits: Optional[torch.FloatTensor] = None,
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

        >>> model = MixtralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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
            # v5's `past_key_values` is a `Cache` object and not subscriptable; the old
            # `past_key_values[0][0].shape[2]` only worked on v4's legacy tuple.
            past_key_values_length = past_key_values.get_seq_length()
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
        if self.config._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # v5 deprecates `_prepare_4d_causal_attention_mask*`; `create_causal_mask` covers both
            # the sdpa and the eager branch. This must follow the actual configured
            # backend, not merely whether FlashAttention is installed.
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                allow_is_causal_skip=False,
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        start_idx, end_idx = stage_index[0], stage_index[1]

        # v5 collects router logits declaratively and activates the collector per call: wrapping only
        # this stage's layer loop in `capture_outputs` yields exactly this stage's logits, which
        # `past_router_logits` then accumulates across stages (as in v4).
        @capture_outputs
        def _local_layers_forward(
            _model,
            hidden_states,
            *,
            output_hidden_states=False,
            output_attentions=False,
            output_router_logits=False,
        ):
            r"""Run this stage's layers. `_model` only exists so that `capture_outputs` can reach
            `_can_record_outputs` and install the hooks; the closure's `self` does the work."""
            local_hidden_states = () if output_hidden_states else None
            local_self_attns = () if output_attentions else None
            next_decoder_cache = None
            for idx, decoder_layer in enumerate(self.layers[start_idx:end_idx], start=start_idx):
                if output_hidden_states:
                    local_hidden_states += (hidden_states,)

                # The cache is updated in place per layer; gradient checkpointing is handled by
                # `GradientCheckpointingLayer.__call__`.
                if self.gradient_checkpointing and self.training:
                    decoder_layer.gradient_checkpointing = True

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
                # v5's `DecoderLayer.forward` returns a bare tensor (v4 returned a tuple)
                hidden_states = layer_outputs

                if use_cache:
                    next_decoder_cache = past_key_values

            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=local_hidden_states,
                attentions=local_self_attns,
            )

        local_outputs = _local_layers_forward(
            self,
            hidden_states,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
        )
        hidden_states = local_outputs.last_hidden_state
        all_hidden_states = local_outputs.hidden_states
        all_self_attns = local_outputs.attentions
        next_decoder_cache = local_outputs.past_key_values
        # `capture_outputs` writes `outputs[key] = tuple(collected)` unconditionally, so "nothing
        # collected" is an empty tuple, not None -- and `load_balancing_loss_func((), ...)` raises
        # `IndexError` on `gate_logits[0]`. Normalise empty to None here; None returns 0 there.
        local_router_logits = tuple(local_outputs.router_logits) if local_outputs.router_logits else None
        all_router_logits = local_router_logits if output_router_logits else None

        if stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        if output_router_logits and past_router_logits is not None:
            # `all_router_logits` is None when this stage collected none (see above).
            all_router_logits = past_router_logits + (all_router_logits or ())

        if stage_manager.is_last_stage():
            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                    if v is not None
                )
            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )
        else:
            if output_router_logits:
                return {
                    "hidden_states": hidden_states,
                    "past_router_logits": all_router_logits,
                }
            else:
                return {
                    "hidden_states": hidden_states,
                }

    @staticmethod
    def mixtral_for_causal_lm_forward(
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        past_router_logits: Optional[torch.FloatTensor] = None,
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

        >>> model = MixtralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

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
        outputs = MixtralPipelineForwards.mixtral_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            past_router_logits=past_router_logits,
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

            # Read by field name, not by position: v5's `ModelOutput` only stores non-None fields,
            # so `outputs[-1]` silently degenerates to `hidden_states` whenever `router_logits` is
            # absent (the field-level read yields the class default `None` instead, which is what
            # `load_balancing_loss_func` wants).
            if output_router_logits:
                # `mixtral_model_forward` returns a tuple when `return_dict=False`; in that
                # form `router_logits` is the final non-None entry (see its tuple construction
                # above).  Field access is only valid for the ModelOutput form.
                router_logits = outputs.router_logits if return_dict else outputs[-1]
            else:
                router_logits = None

            aux_loss = None
            if output_router_logits:
                aux_loss = load_balancing_loss_func(router_logits, self.num_experts, self.num_experts_per_tok)
                if labels is not None:
                    loss += self.router_aux_loss_coef * aux_loss

            if not return_dict:
                output = (logits,) + outputs[1:]
                if output_router_logits:
                    output = (aux_loss,) + output
                return (loss,) + output if loss is not None else output

            return MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss,
                logits=logits,
                past_key_values=None,
                hidden_states=outputs[0],
                attentions=None,
                router_logits=router_logits,
            )
        else:
            out = {}
            hidden_states = outputs.get("hidden_states")
            out["hidden_states"] = hidden_states
            if output_router_logits:
                out["past_router_logits"] = outputs["past_router_logits"]
            return out


def get_mixtral_flash_attention_forward(shard_config, sp_mode=None, sp_size=None, sp_group=None):
    logger = logging.get_logger(__name__)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.mixtral.modeling_mixtral import eager_attention_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # Keep the plural name: v5's call sites pass `past_key_values=`, so a singular parameter
        # would swallow it into `**kwargs` and silently disable the KV cache.
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if sp_mode is not None:
            assert sp_mode in ["all_to_all", "split_gather", "ring"], "Invalid sp_mode"
            assert (sp_size is not None) and (
                sp_group is not None
            ), "Must specify sp_size and sp_group for sequence parallel"

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()
        # v5's `MixtralAttention` has no `num_heads` / `num_key_value_heads` attributes of its own
        # and reads the config directly. Only the TP and SP-all_to_all policies write those names,
        # so a plain SP run would hit an `AttributeError` on `self.num_heads` -- hence the fallback.
        num_heads = getattr(self, "num_heads", None) or self.config.num_attention_heads
        num_key_value_heads = getattr(self, "num_key_value_heads", None) or self.config.num_key_value_heads

        # sp: modify sp_len when sequence parallel mode is ring
        if sp_mode in ["split_gather", "ring"]:
            q_len *= sp_size

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            query_states = all_to_all_comm(query_states, sp_group, fp8_communication=shard_config.fp8_communication)
            key_states = all_to_all_comm(key_states, sp_group, fp8_communication=shard_config.fp8_communication)
            value_states = all_to_all_comm(value_states, sp_group, fp8_communication=shard_config.fp8_communication)
            bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        if past_key_values is not None and self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )
        if past_key_values is not None:
            # v5 dropped `Cache.get_usable_length`; `update` now returns the full (history
            # included) k/v, matching v5's own `MixtralAttention`.
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # v5's `eager_attention_forward` / `sdpa_attention_forward` both do their own `repeat_kv`
        # (keyed on `num_key_value_groups`) and expect q/k/v in `[bsz, heads, q_len, head_dim]`.
        # The v4 leftovers here -- an explicit `repeat_kv` plus a `transpose(1, 2)` before the
        # interface -- double-count the head dim and shift the sequence into the head axis.
        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )

        # sp: all-to-all comminucation when introducing sequence parallel
        if sp_mode == "all_to_all":
            attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim).contiguous()  # (1, 8, 128)
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2, fp8_communication=shard_config.fp8_communication
            )  # (1, 4, 256)
        else:
            attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    return forward


def get_mixtral_flash_attention_model_forward(shard_config, sp_mode=None, sp_size=None, sp_group=None):
    logger = logging.get_logger(__name__)

    # v5 collects `router_logits` declaratively (`OutputRecorder(MixtralTopKRouter, index=0)` in
    # `MixtralModel._can_record_outputs`) and injects it via `@capture_outputs`. v5's own forward
    # carries that decorator and ours replaces it, so it has to be put back here.
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if (self.gradient_checkpointing or sp_mode in ["ring", "all_to_all"]) and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        if use_cache:
            # v5 dropped `DynamicCache.from_legacy_cache` / `to_legacy_cache`, but the
            # `ddp_cache_data` argument of `DynamicCache.__init__` is exactly the per-layer
            # `(key, value[, sliding_window])` tuples, so constructing it is the old semantics.
            if past_key_values is None:
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache(past_key_values)
            # v5 dropped `Cache.get_usable_length` in favour of `get_seq_length`.
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self.config._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        if self.config._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # v5 deprecates `modeling_attn_mask_utils._prepare_4d_causal_attention_mask*` in
            # favour of `create_causal_mask`, which also folds in the sdpa / eager split and
            # sliding window. `allow_is_causal_skip=False` because this branch ends in explicit
            # softmax rather than delegating causality to the backend.
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                allow_is_causal_skip=False,
            )

        if sp_mode in ["ring", "split_gather"]:
            inputs_embeds = split_forward_gather_backward(
                inputs_embeds, 1, sp_group, fp8_communication=shard_config.fp8_communication
            )
        elif sp_mode == "all_to_all":
            inputs_embeds = split_forward_gather_backward(
                inputs_embeds, 1, sp_group, 1 / sp_size, fp8_communication=shard_config.fp8_communication
            )
        hidden_states = inputs_embeds

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # v5 hands gradient checkpointing to `GradientCheckpointingLayer.__call__`, but
            # `_gradient_checkpointing_func` is still required (it is called there) and is only
            # installed by going through `gradient_checkpointing_enable()` -- hence the flag.
            if self.gradient_checkpointing and self.training:
                decoder_layer.gradient_checkpointing = True

            # Keyword names follow v5's `MixtralDecoderLayer.forward`. The old positional call
            # fails outright (7 args against 6 parameters), and would misalign `position_embeddings`
            # even if the count matched, since v5 puts it second.
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

            # v5's `DecoderLayer.forward` returns a bare tensor (v4 returned a tuple)
            hidden_states = layer_outputs

            if use_cache:
                # v5 updates the passed-in `Cache` in place instead of returning it
                next_decoder_cache = past_key_values

        hidden_states = self.norm(hidden_states)

        if sp_mode == "ring" or sp_mode == "split_gather":
            hidden_states = gather_forward_split_backward(
                hidden_states, 1, sp_group, fp8_communication=shard_config.fp8_communication
            )
        elif sp_mode == "all_to_all":
            hidden_states = gather_forward_split_backward(
                hidden_states, 1, sp_group, grad_scale=sp_size, fp8_communication=shard_config.fp8_communication
            )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            # v5 dropped `to_legacy_cache`: the cache comes back as a `Cache` object, as v5
            # itself does, no longer downgraded to a legacy tuple to match the input.
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            # Left to the `@capture_outputs` decorator (see the top of this function).
            router_logits=None,
        )

    return forward

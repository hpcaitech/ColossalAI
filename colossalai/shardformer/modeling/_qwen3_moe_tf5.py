r"""Qwen3-MoE sharding helpers for transformers v5.

``Qwen3MoeEPExperts`` turns the fused ``Qwen3MoeExperts`` of a sparse block into the shared
:class:`~colossalai.moe.EPMoELayer`. The router and the rest of the block stay the native HF modules: v5 calls
``self.experts(hidden_states, top_k_index, top_k_weights)``, which is exactly the signature of the layer, and the layer
keeps the parameter names ``gate_up_proj`` / ``down_proj``.
"""

import warnings

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from colossalai.lazy import LazyInitContext
from colossalai.moe import EPMoELayer, GroupedMMExperts, ReferenceExperts
from colossalai.tensor.moe_tensor.api import set_expert_grad_scale, set_moe_tensor_ep_fused, set_moe_tensor_ep_group

__all__ = ["Qwen3MoeEPExperts"]


def _shard_param_in_place(param: torch.nn.Parameter, local: torch.Tensor) -> None:
    """Swap a parameter's storage for its local shard while keeping the Parameter object.

    Optimizers are built from the unsharded model and the plugins only prune (never re-add) optimizer params, so
    replacing the Parameter object would silently drop the experts from training.
    """
    param.data = local


class Qwen3MoeEPExperts:
    r"""Factory used by the :class:`Sharder`: ``from_native_module`` returns an ``EPMoELayer``.

    The returned layer holds the *original* ``gate_up_proj`` / ``down_proj`` Parameter objects, whose storage is swapped
    for this rank's experts in place (see :func:`_shard_param_in_place`).
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(f"Please use `from_native_module` to create an instance of {self.__class__.__name__}")

    @staticmethod
    def from_native_module(
        module: Qwen3MoeExperts,
        process_group: ProcessGroup = None,
        ep_group: ProcessGroup = None,
        **kwargs,
    ) -> EPMoELayer:
        del process_group  # the tensor parallel group: the experts are not tensor parallel
        assert ep_group is not None, "Qwen3MoeEPExperts needs the expert parallel group"
        LazyInitContext.materialize(module)

        try:
            experts = GroupedMMExperts()
        except RuntimeError:  # torch < 2.8 has no torch._grouped_mm
            warnings.warn("torch._grouped_mm is not available, falling back to the slower per-expert loop.")
            experts = ReferenceExperts()
        # built on the meta device only to get the dispatcher, the backend and the bookkeeping (which experts this rank
        # holds); its parameters are replaced by the (sharded) original ones below
        layer = EPMoELayer(
            module.num_experts,
            module.hidden_dim,
            module.intermediate_dim,
            ep_group=ep_group,
            act_fn=module.act_fn,
            experts=experts,
            device="meta",
        )

        gate_up_proj, down_proj = module.gate_up_proj, module.down_proj
        local = layer.local_expert_range
        _shard_param_in_place(gate_up_proj, gate_up_proj.data[local.start : local.stop].clone())
        _shard_param_in_place(down_proj, down_proj.data[local.start : local.stop].clone())
        layer.gate_up_proj = gate_up_proj
        layer.down_proj = down_proj
        for param in (gate_up_proj, down_proj):
            set_moe_tensor_ep_group(param, ep_group)
            set_expert_grad_scale(param, layer.expert_grad_scale)
            # checkpoints: gather the experts of all EP ranks when saving (MoECheckpointIO); the layer's own load hook
            # slices a full ``[num_experts, ...]`` tensor down to this rank's experts and reads the parameters at call
            # time, so it also serves the original Parameter objects assigned above
            set_moe_tensor_ep_fused(param)
        return layer

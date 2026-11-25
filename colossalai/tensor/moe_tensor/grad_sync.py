"""Gradient synchronization plan of the MoE plugin.

The model adaptation layer only tags what a parameter is: an expert or not and, for an expert, where its ``1 / ep_size``
factor is applied (:class:`~colossalai.tensor.moe_tensor.api.ExpertGradScale`). The plugin combines those tags with the
process groups of the parallel topology into one :class:`GradSyncSpec` per parameter, and every reducer reads that
plan instead of classifying parameters on its own.

The contract is ``final grad = SUM(grad over replica_group) / sum_divisor``. The size of the group and the divisor are
different numbers: an expert whose layer leaves ``1 / ep_size`` to the reduction is summed over the ``moe_dp`` group but
divided by ``moe_dp_size * ep_size``.
"""

from dataclasses import dataclass
from typing import Dict, Iterable

import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from .api import ExpertGradScale, get_ep_size, get_expert_grad_scale, is_moe_tensor

__all__ = ["GradSyncSpec", "build_grad_sync_plan", "has_reduction_scaled_experts"]


@dataclass(frozen=True)
class GradSyncSpec:
    """How the gradient of one parameter is reduced.

    Args:
        replica_group: the ranks that hold replicas of the parameter, i.e. where the gradients are summed.
        sum_divisor: the number the sum is divided by.
    """

    replica_group: ProcessGroup
    sum_divisor: float

    @property
    def extra_divisor(self) -> float:
        """What is left to divide by once the gradient has been averaged over ``replica_group``."""
        return self.sum_divisor / dist.get_world_size(self.replica_group)


def build_grad_sync_plan(
    params: Iterable[nn.Parameter], dp_group: ProcessGroup, moe_dp_group: ProcessGroup
) -> Dict[nn.Parameter, GradSyncSpec]:
    """Map each parameter to its :class:`GradSyncSpec`.

    Args:
        params: the parameters of the model, after the model has been sharded and moved to its device.
        dp_group: replica group of the dense parameters, the router and the shared experts (``mixed_dp_group``).
        moe_dp_group: replica group of the routed experts (``mixed_moe_dp_group``).
    """
    dp_size = dist.get_world_size(dp_group)
    moe_dp_size = dist.get_world_size(moe_dp_group)
    plan = {}
    for param in params:
        if not is_moe_tensor(param):
            plan[param] = GradSyncSpec(dp_group, dp_size)
        elif get_expert_grad_scale(param) == ExpertGradScale.IN_REDUCTION:
            plan[param] = GradSyncSpec(moe_dp_group, moe_dp_size * get_ep_size(param))
        else:  # IN_FORWARD (also the default of untagged experts): the layer already applied 1 / ep_size
            plan[param] = GradSyncSpec(moe_dp_group, moe_dp_size)
    return plan


def has_reduction_scaled_experts(params: Iterable[nn.Parameter]) -> bool:
    """Whether some parameter leaves the ``1 / ep_size`` of its gradient to the gradient reduction."""
    return any(is_moe_tensor(p) and get_expert_grad_scale(p) == ExpertGradScale.IN_REDUCTION for p in params)

# adapted from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/adamw8bit.py
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
from bitsandbytes.optim.optimizer import Optimizer2State

from colossalai.interface.optimizer import DistributedOptim
from colossalai.tensor.d_tensor import is_distributed_tensor
from colossalai.tensor.d_tensor.sharding_spec import DimSpec

from .galore import GaLoreProjector

__all__ = ["DistributedGalore"]
# Mark sharded dimension
_SHARD_DIM = DimSpec([0])


def get_shard_dim(p):
    if not is_distributed_tensor(p):
        raise ValueError("p is not a distributed tensor")
    sharding = p.dist_layout.sharding_spec.sharding_sequence
    return sharding.index(_SHARD_DIM)


class DistGaloreAwamW8bit(DistributedOptim, Optimizer2State):
    r"""Implements Galore, a optimizer-agonistic gradient compression technique on 8-bit AdamW.
    It largely compresses gradient via low-rank projection and is claimed to be insensitive to hyperparams like lr.
    Supports Tensor Parallel and ZeRO stage 1 and 2 via setup_distributed inside booster and plugin.
    Proposed in `GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection`
    https://arxiv.org/abs/2403.03507

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
    Example:
        >>> optim = DistributedLamb(model.parameters(), lr=1e-3)
        >>> proc_mesh = ProcessGroupMesh(tp_size, zero_size)
        >>> tp_group = proc_mesh.get_group_along_axis(0)
        >>> dp_group = proc_mesh.get_group_along_axis(1)
        >>> optim.setup_distributed(tp_group, dp_group)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
    ):
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,
            args,
            min_8bit_size,
            percentile_clipping,
            block_wise,
            is_paged=is_paged,
        )
        self.tp_size = 0
        self.dp_size = 0
        self.is_dist = {}
        proj_none = all(["rank" not in group for group in self.param_groups])
        if proj_none:
            print(
                "Will not apply GaLore as no rank is specified. Or did you forget to?\
                Try get_galore_param_groups(model)"
            )

        # Default from the paper
        for group in self.param_groups:
            if "rank" in group:
                group["update_proj_gap"] = group.get("update_proj_gap", 200)
                group["proj_type"] = group.get("proj_type", "std")
                group["scale"] = group.get("scale", 0.25)

    def setup_distributed(
        self,
        tp_group: Optional[dist.ProcessGroup] = None,
        dp_group: Optional[dist.ProcessGroup] = None,
        shard_to_working_param: Optional[Dict] = {},
        padding_map: Optional[Dict] = defaultdict(int),
        is_zero: Optional[bool] = False,
    ):
        """Assign process groups for TP and ZeRO 2.
        Arguments:
            tp_group (dist.ProcessGroup): Tensor Parallel process group
            dp_group (dist.ProcessGroup): ZeRO 2 process group
            shard_to_working_param (Dict): ZeRO 2 feeds the optimizer a sharded param view to match grad shape.
                This maps from id(view) to model params used in forward & backward.
            padding_map (Dict): Padding size of each param from ZeRO's param store. Required if ZeRO is used.
            is_zero (bool): Whether to use ZeRO 2.
        """
        assert dist.is_initialized(), "You forgot to initialized distributed backend..."

        self.tp_group = tp_group
        self.dp_group = dp_group
        if tp_group is not None:
            self.tp_size = dist.get_world_size(tp_group)
        if dp_group is not None:
            self.dp_size = dist.get_world_size(dp_group)

        self.shard_to_working_param = shard_to_working_param if shard_to_working_param is not None else {}
        self.is_zero = is_zero
        if is_zero:
            assert padding_map is not defaultdict(0), "We can't do SVD without knowing ZeRO's per-param padding size"
        self.padding_map = padding_map
        self.distributed_on = self.tp_size > 0 or self.dp_size > 0

        # Cache working param layout
        self.shard_dim = {}
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in self.shard_to_working_param:
                    # No ZeRO; master param = working param
                    self.shard_to_working_param[id(p)] = p

                self.is_dist[id(p)] = is_distributed_tensor(self.shard_to_working_param[id(p)])
                if is_distributed_tensor(self.shard_to_working_param[id(p)]):
                    self.shard_dim[id(p)] = get_shard_dim(self.shard_to_working_param[id(p)])

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjector(
                            group["rank"],
                            scale=group["scale"],
                            update_proj_gap=group["update_proj_gap"],
                            proj_type=group["proj_type"],
                        )

                    if "weight_decay" in group and group["weight_decay"] > 0:
                        # ensure that the weight decay is not applied to the norm grad
                        group["weight_decay_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    grad = p.grad
                    working_shape = self.shard_to_working_param[id(p)].shape
                    # Restore unsharded working param shape for projection
                    if self.distributed_on:
                        # Both the ZeRO 1 & 2 implementation don't retain full grads
                        if self.dp_size > 1 and self.is_zero:
                            grad = grad[: self.padding_map[id(p)]]  # unpad
                            split_factor = self.dp_size

                            # Gather grads for projection
                            if state["step"] % group["update_proj_gap"] == 0:
                                all_grads = [
                                    torch.empty_like(grad, dtype=p.grad.dtype, device=p.grad.device)
                                    for _ in range(self.dp_size)
                                ]
                                dist.all_gather(all_grads, grad, self.dp_group)
                                grad = torch.cat(all_grads)
                                split_factor = 1

                            # For correct reshaping both w/ and w/o ZeRO:
                            # (m, n).flatten().chunk(dp_size) equals to
                            # (m / dp_size, n).flatten()
                            working_shape[0] /= split_factor
                            grad = grad.reshape(working_shape)

                        if self.is_dist[id(p)] and state["step"] % group["update_proj_gap"] == 0:
                            all_grads = [
                                torch.empty_like(grad, dtype=p.grad.dtype, device=p.grad.device)
                                for _ in range(self.tp_size)
                            ]
                            dist.all_gather(all_grads, grad, self.tp_group)
                            grad = torch.cat(all_grads, dim=self.shard_dim[id(p)])

                        # Compute SVD. Will adaptively use a subset of singular
                        # vectors during update_proj_gap when grads are not gathered.
                        grad = state["projector"].project(grad, state["step"])

                        # Post-projection sharding to master shape
                        if self.distributed_on:
                            if self.is_dist[id(p)] and state["step"] % group["update_proj_gap"] == 0:
                                grad = grad.chunk(self.tp_size, dim=self.shard_dim[id(p)])[dist.get_rank(self.tp_group)]

                            if self.dp_size > 1 and self.is_zero:
                                if state["step"] % group["update_proj_gap"] == 0:
                                    grad = grad.flatten().chunk(self.dp_size)[dist.get_rank(self.dp_group)]

                        # pad back to master param shape
                        grad = torch.nn.functional.pad(grad, [0, self.padding_map[id(p)]])
                        assert grad.shape == p.shape

                    # suboptimal implementation
                    p.saved_data = p.data.clone()
                    p.data = grad.clone().to(p.data.dtype).to(p.data.device)
                    p.data.zero_()
                    p.grad = grad

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()

                # GaLore Projection Back
                if "rank" in group:
                    data = p.data[: -self.padding_map[id(p)]].view(working_shape)  # to working shape
                    data = state["projector"].project_back(data).view(-1)  # to master shape
                    data = torch.pad(data, [0, self.padding_map[id(p)]])
                    p.data = p.saved_data.add_(data)

                    # apply decoupled weight decay
                    if "weight_decay_saved" in group:
                        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay_saved"])
                        group["weight_decay"] = group["weight_decay_saved"]
                        del group["weight_decay_saved"]

        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()
        return loss

""" adapted from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/adamw8bit.py"""

import warnings
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from bitsandbytes.optim.optimizer import Optimizer2State

from colossalai.interface.optimizer import DistributedOptim
from colossalai.tensor.d_tensor import get_shard_dim_1d, is_distributed_tensor

from .galore import GaLoreProjector, make_low_rank_buffer

__all__ = ["DistributedGalore"]
# Mark sharded dimension


class DistGaloreAwamW(DistributedOptim, Optimizer2State):
    r"""Implements Galore, a optimizer-agonistic gradient compression technique on 8-bit AdamW.
    It largely compresses gradient via low-rank projection and is claimed to be insensitive to hyperparams like lr.
    Supports Tensor Parallel and ZeRO stage 1 and 2 via booster and plugin.
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
        nbits: Number of bits for quantization optim states. Only 32 and 8 are supported.
        min_8bit_size (`int`, defaults to 4096):
            The minimum number of elements of the parameter tensors for 8-bit optimization.
        percentile_clipping (`int`, defaults to 100):
            Adapts clipping threshold automatically by tracking the last 100 gradient norms and clipping the gradient at a certain percentile to improve stability.
        block_wise (`bool`, defaults to `True`):
            Whether to independently quantize each block of tensors to reduce outlier effects and improve stability.
        is_paged (`bool`, defaults to `False`):
            Whether the optimizer is a paged optimizer (handle memory spike via CPU-GPU transfer) or not.
        args (dict, optional): quantization-related arguments. If passed, will override all quantization args above.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        nbits=8,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
        args=None,
    ):
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits=nbits,
            args=args,
            min_8bit_size=min_8bit_size,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged,
        )

        self.tp_size = 1
        self.dp_size = 1
        self.is_dist = {}
        proj_none = all(["rank" not in group for group in self.param_groups])
        if proj_none:
            warnings.warn(
                "Will not apply GaLore as rank isn't in any param group. If you forgot to, try get_galore_param_groups"
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
        """Setup process groups for TP and ZeRO 2.
        Arguments:
            tp_group (dist.ProcessGroup): Tensor Parallel process group
            dp_group (dist.ProcessGroup): ZeRO 2 process group
            shard_to_working_param (Dict): ZeRO 2 feeds the optimizer a sharded param view as grads are sharded.
                This maps from id(view) to working params used in forward & backward.
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
        self.is_zero = is_zero and self.dp_size > 1
        self.padding_map = padding_map if padding_map is not None else defaultdict(int)
        if is_zero:
            assert self.padding_map is not defaultdict(
                int
            ), "We can't do SVD without knowing ZeRO's per-param padding size"
        self.distributed_on = self.tp_size > 0 or self.dp_size > 0

        # Cache working param layout
        self.shard_dim = {}
        for group in self.param_groups:
            for p in group["params"]:
                # w/o ZeRO: master param = working param
                self.shard_to_working_param[id(p)] = self.shard_to_working_param.get(id(p), p)
                if id(p) not in self.padding_map:
                    self.padding_map[id(p)] = 0

                self.is_dist[id(p)] = is_distributed_tensor(self.shard_to_working_param[id(p)])
                if is_distributed_tensor(self.shard_to_working_param[id(p)]):
                    self.shard_dim[id(p)] = get_shard_dim_1d(self.shard_to_working_param[id(p)])

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
                    # decoupled weight decay
                    if "weight_decay" in group and group["weight_decay"] > 0:
                        group["weight_decay_saved"] = group["weight_decay"]
                        group["weight_decay"] = 0

                    grad = p.grad
                    working_shape = list(self.shard_to_working_param[id(p)].shape)
                    padding = self.padding_map[id(p)]

                    # All-gather grads for projection step
                    if self.distributed_on:
                        # Gather for ZeRO 1 & 2 implementation don't retain full grads
                        if self.is_zero:
                            # (m, n).flatten().chunk(dp_size) equals to (m / dp_size, n).flatten()
                            working_shape[0] //= self.dp_size
                            # Gather grads for projection
                            if state["step"] % group["update_proj_gap"] == 0:
                                all_grads = [
                                    torch.empty_like(grad, dtype=p.grad.dtype, device=p.grad.device)
                                    for _ in range(self.dp_size)
                                ]
                                dist.all_gather(all_grads, grad, self.dp_group)
                                grad = torch.cat(all_grads)
                                # To working param shape
                                if padding > 0:
                                    grad = grad[:-padding]
                                working_shape[0] *= self.dp_size
                            grad = grad.reshape(working_shape)  # unflatten

                        # Gather TP grads
                        if self.is_dist[id(p)] and state["step"] % group["update_proj_gap"] == 0:
                            all_grads = [
                                torch.empty_like(grad, dtype=p.grad.dtype, device=p.grad.device)
                                for _ in range(self.tp_size)
                            ]
                            dist.all_gather(all_grads, grad.contiguous(), self.tp_group)
                            grad = torch.cat(all_grads, dim=self.shard_dim[id(p)])

                    # Compute SVD. Will use a subset of singular vectors when grads are sharded.
                    grad = state["projector"].project(grad, state["step"])

                    # Re-shard gathered grads after SVD
                    if self.distributed_on and state["step"] % group["update_proj_gap"] == 0:
                        # TP
                        if self.is_dist[id(p)]:
                            grad = grad.chunk(self.tp_size, dim=self.shard_dim[id(p)])[dist.get_rank(self.tp_group)]
                        # ZeRO
                        # TODO: this might not work with padding, e.g. (3, 3) with dp size 2
                        # Need extra logic in ZeRO to pad nRows/nCols to be divisible by dp_size
                        if self.is_zero:
                            grad = grad.chunk(self.dp_size)[dist.get_rank(self.dp_group)]
                        grad = grad.contiguous()  # avoid bitsandbytes update error

                    working_shape = grad.shape
                    # To flattended master param shape
                    grad = self.to_master_shape(grad, padding)
                    make_low_rank_buffer(p, grad)

                if "state1" not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()

                # Project Back to working param shape
                if "rank" in group:
                    # Unpad
                    if self.is_zero:
                        if padding > 0:
                            p.data = p.data[:-padding]
                        p.data = p.data.reshape(working_shape)

                    p.data = state["projector"].project_back(p.data)
                    # Re-flatten grads for ZeRO
                    p.data = self.to_master_shape(p.data, padding)
                    p.data = p.saved_data.add_(p.data)

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

    def to_master_shape(self, data, padding):
        """Pad to master (optimizer) param shape"""
        if not self.is_zero:
            return data
        data = data.view(-1)
        if padding > 0:
            data = F.pad(data, [0, padding])
        return data

    def __del__(self):
        """Avoid buffer memory leak"""
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "saved_data"):
                    del p.saved_data

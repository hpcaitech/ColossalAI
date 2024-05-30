# Disclaimer: Modified from https://github.com/NUS-HPC-AI-Lab/pytorch-lamb/blob/master/optim/lamb.py


from typing import Dict, Optional

import torch
import torch.distributed as dist

from colossalai.interface.optimizer import DistributedOptim
from colossalai.tensor.d_tensor import is_distributed_tensor

__all__ = ["DistributedLamb"]


class DistributedLamb(DistributedOptim):
    r"""Implements the Lamb algorithm, with extra support for ZeRO 2 and Tensor Parallel.
    Proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    It's recommended to use this with HybridParallelPlugin/ZeRO plugin and booster,
    which will take care of setup_distributed.
    Example with 4 devices:
        >>> optim = DistributedLamb(model.parameters(), lr=1e-3)
        >>> proc_mesh = ProcessGroupMesh(tp_size, zero_size)
        >>> tp_group = proc_mesh.get_group_along_axis(0)
        >>> dp_group = proc_mesh.get_group_along_axis(1)
        >>> optim.setup_distributed(tp_group, dp_group)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        bias_correction=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # self.setup_distributed(tp_group, dp_group)
        self.shard_to_working_param = {}
        self.tp_size = self.dp_size = 1
        self.is_zero = False
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    def setup_distributed(
        self,
        tp_group: Optional[dist.ProcessGroup] = None,
        dp_group: Optional[dist.ProcessGroup] = None,
        shard_to_working_param: Optional[Dict] = {},
        padding_map=None,
        is_zero: Optional[bool] = False,
    ):
        """Assign process groups for TP and ZeRO 2.
        Arguments:
            tp_group (dist.ProcessGroup): Tensor Parallel process group
            dp_group (dist.ProcessGroup): ZeRO 2 process group
            shard_to_working_param (Dict): ZeRO 2 feeds the optimizer a sharded param view as grads are sharded.
                This maps from id(view) to working params used in forward & backward.
            padding_map: An empty interface placeholder
            is_zero (bool): Whether to use ZeRO 2.
        """
        self.tp_group = tp_group
        self.dp_group = dp_group
        if tp_group is not None:
            self.tp_size = dist.get_world_size(tp_group)
        if dp_group is not None:
            self.dp_size = dist.get_world_size(dp_group)

        self.shard_to_working_param = shard_to_working_param if shard_to_working_param is not None else {}
        self.is_zero = is_zero
        self.is_dist = {}
        # Cache parameter layout
        for group in self.param_groups:
            for p in group["params"]:
                # w/o ZeRO: master param = working param
                self.shard_to_working_param[id(p)] = self.shard_to_working_param.get(id(p), p)
                self.is_dist[p] = (
                    is_distributed_tensor(p)
                    if self.dp_size <= 1
                    else is_distributed_tensor(self.shard_to_working_param.get(id(p), None))
                )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instad.")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                scaled_lr = group["lr"]
                if group["bias_correction"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    # Apply debiasing to lr to avoid broadcast
                    scaled_lr *= (bias_correction2**0.5) / bias_correction1
                    # exp_avg.div_(bias_correction1)
                    # exp_avg_sq.div_(bias_correction2)

                update = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                # Compute global layer-wise trust ratio
                if self.is_dist[p] or self.is_zero:
                    p_local = p
                    g_sum = (update**2).sum()
                    if self.dp_size > 1 and self.is_zero:
                        # ZeRO 2 doesn't shard param. Compute full param norm w/o communication.
                        dist.all_reduce(g_sum, group=self.dp_group)
                        p_local = self.shard_to_working_param[id(p)]

                    w_sum = (p_local**2).sum()
                    sums = torch.stack([w_sum, g_sum])

                    # Get global l2 norms
                    if self.tp_size > 1:
                        dist.all_reduce(sums, group=self.tp_group)
                    w_norm, g_norm = sums.sqrt().chunk(2)
                else:
                    # Fall back to vanilla Lamb
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(update)

                trust_ratio = torch.where(w_norm > 0 and g_norm > 0, (w_norm / g_norm), 1.0).item()

                scaled_lr *= trust_ratio
                p.data.add_(update, alpha=-scaled_lr)

        return loss

# Disclaimer: Modified from https://github.com/NUS-HPC-AI-Lab/pytorch-lamb/blob/master/optim/lamb.py

import warnings

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from colossalai.device.device_mesh import DeviceMesh

__all__ = ["DistributedLamb"]


class DistributedLamb(Optimizer):
    r"""Implements the Lamb algorithm, with extra support for ZeRO 2 and Tensor Parallel.
    Proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
        device_mesh: a 2D device mesh containing process groups for TP and ZeRO 2, initialized
            from init_distributed() method.
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
        adam=False,
        bias_correction=True,
        device_mesh=None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.adam = adam
        self.device_mesh = device_mesh
        self.tp_group = device_mesh.get_process_group(axis=0) if device_mesh is not None else None
        self.dp_group = device_mesh.get_process_group(axis=1) if device_mesh is not None else None

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    @staticmethod
    def init_distributed(tp_size, zero_size):
        """Initializes a device mesh containing process groups for TP and ZeRO 2"""
        world_size = tp_size * zero_size
        os_world_size = dist.get_world_size(None)
        if world_size == 1:
            warnings.warn(
                "You are using single device training. Distributed Lamb is not necessary\
                and won't be initialized."
            )
        else:
            device_ids = torch.arange(world_size)
            mesh_shape = torch.Size((tp_size, zero_size))
            assert (
                os_world_size == world_size
            ), f"You launched {os_world_size} processes != tp_size * dp_size = {world_size}"
            device_mesh = DeviceMesh(device_ids, mesh_shape, init_process_group=True)
            tp_group = device_mesh.get_process_group(axis=0)
            dp_group = device_mesh.get_process_group(axis=1)
            return device_mesh, tp_group, dp_group

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

        torch.nn.utils.clip_grad_norm_(
            parameters=[p for group in self.param_groups for p in group["params"]], max_norm=1.0, norm_type=2
        )

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

                if not self.adam:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(update)
                    # TODO: Call all-reduce here
                    if self.device_mesh is not None:
                        norms = torch.stack([w_norm, g_norm])
                        if self.dp_group is not None:
                            dist.all_reduce(norms, group=self.dp_group)
                        if self.tp_group is not None:
                            dist.all_reduce(norms, group=self.tp_group)

                        # No need to average the norms as we compute their ratio.
                        w_norm, g_norm = norms.chunk(2)

                    trust_ratio = torch.where(w_norm > 0 and g_norm > 0, w_norm / g_norm, torch.ones_like(w_norm))
                    scaled_lr *= trust_ratio.item()

                p.data.add_(update, alpha=-scaled_lr)

        return loss

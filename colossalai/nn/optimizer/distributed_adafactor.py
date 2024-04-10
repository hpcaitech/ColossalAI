import math
from typing import Dict

import torch
import torch.distributed as dist
# from torch.optim import Optimizer
from colossalai.interface.optimizer import DistributedOptim

from colossalai.shardformer.layer._operation import _gather, _split
from colossalai.tensor.d_tensor import get_sharding_spec, is_distributed_tensor

# DistributedAdaFactor (with Tensor parallel and Zero stage 2)
__all__ = ["DistributedAdaFactor"]


class DistributedAdaFactor(DistributedOptim):
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        lr=None
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        self.tensor_parallel_size = 1
        self.tensor_parallel_group = None
        self.data_parallel_size = 1
        self.data_parallel_group = None
        self.shard_to_param = None  # Dict{id:shape}, sample {id(param): torch.tensor}
        self.shard_spec = None
        self.grad_shape = None
        self.factored = None  # bool
        self.use_first_moment = None  # bool
        self.use_zero = True
        super().__init__(params, defaults)
        

    def setup_distributed(
        self,
        tensor_parallel_group: dist.ProcessGroup = None,
        data_parallel_group: dist.ProcessGroup = None,
        shard_to_param: Dict = {},
        use_zero: bool = True,
    ) -> None:
        """
        Inject features to the Optimizer

        Args:
            tensor_parallel_group: The devices group for tensor parallel;
            data_parallel_group: The devices group for data parallel;
            sharding_spec_dict: ShardingSpecs of Each params;
            param_shape: Paramater Shape of Each params;
            use_zero: Whether or not to use zero;

        """
        self.tensor_parallel_group = tensor_parallel_group  # "Expected row process group"
        self.data_parallel_group = data_parallel_group
        if self.tensor_parallel_group is not None:
            self.tensor_parallel_size = dist.get_world_size(self.tensor_parallel_group)
        if self.data_parallel_group is not None:
            self.data_parallel_size = dist.get_world_size(self.data_parallel_group)
        self.use_zero = use_zero
        
        self.shard_to_param = shard_to_param if shard_to_param is not None else {}
        

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        """
        Determines whether the current param is factored
        Args:
            param_group : param group
            param_shape : Original Shape of param

        """
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    # approx_sq_grad for row parallel weight
    @staticmethod
    def _approx_sq_grad_row_parallel(exp_avg_sq_row, exp_avg_sq_col, sq_row_meam):
        # row_meam = sq_row_meam
        r_factor = (exp_avg_sq_row / sq_row_meam).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization steps
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        """
        param_groups: Dict
        {
            "params":[weight, bias]
            "lr"
            "eps"
            "clip_threshold"
            "decay_rate"
            "beta1"
            "weight_decay"
            "scale_parameter"
            "relative_step"
            "warmup_init"
        }
        """
        for group in self.param_groups:
            # update weight & bias
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                state = self.state[p]
                self.grad_shape = grad.shape  # 1 dim shape
                
                # print(f"self.shard_to_param {self.shard_to_param}")
                
                param_is_dtensor = is_distributed_tensor(self.shard_to_param.get(id(p)))

                if param_is_dtensor:
                    self.grad_shape = self.shard_to_param.get(id(p)).shape  # tp shape (2 dim)

                self.factored, self.use_first_moment = self._get_options(group, self.grad_shape)
                if len(state) == 0:
                    state["step"] = 0
                    if self.use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                    if self.factored:
                        self.shard_spec = get_sharding_spec(self.shard_to_param.get(id(p)))
                        if self.shard_spec.sharding_sequence[0] == "R":  # Col Parallel
                            state["exp_avg_sq_row"] = torch.zeros(
                                self.grad_shape[0] // self.data_parallel_size, device=p.device, dtype=p.dtype
                            )  # [H/dp]
                            state["exp_avg_sq_col"] = torch.zeros(
                                self.grad_shape[1], device=p.device, dtype=p.dtype
                            )  # [W/TP]

                        if self.shard_spec.sharding_sequence[-1] == "R":  # Row Parallel
                            # Row indivisible shape situation
                            if self.grad_shape[0] % self.data_parallel_size != 0:
                                state["exp_avg_sq_row"] = torch.zeros(
                                self.grad_shape[0], device=p.device, dtype=p.dtype
                                )  # [H/dp/Tp]
                            else:
                                state["exp_avg_sq_row"] = torch.zeros(
                                self.grad_shape[0] // self.data_parallel_size, device=p.device, dtype=p.dtype
                                )  # [H/dp/Tp]
                            
                            state["exp_avg_sq_col"] = torch.zeros(
                                self.grad_shape[1], device=p.device, dtype=p.dtype
                            )  # [W]
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["RMS"] = 0
                else:
                    if self.use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if self.factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                state["step"] += 1
                lr = self._get_lr(group, state)
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]

                if self.factored:
                    # ==============================
                    # First Dim is R, Last Dim is S{} means split dim -1  --->
                    # Coloum Parallel ---> sq_row need Do (col) Reduce
                    # ==============================
                    self.shard_spec = get_sharding_spec(self.shard_to_param.get(id(p)))
                    if self.shard_spec.sharding_sequence[0] == "R":
                        update_reshape = update.view(-1, self.grad_shape[1])
                        grad_reshape = grad.view(-1, self.grad_shape[1])
                        exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/dp]
                        exp_avg_sq_col = state["exp_avg_sq_col"]  # [W/tp]
                        exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                        exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                        dist.all_reduce(exp_avg_sq_row, group=self.tensor_parallel_group)
                        exp_avg_sq_row.div_(self.tensor_parallel_size)
                        update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                        update_reshape.mul_(grad_reshape)
                        if self.use_zero:
                            update = update_reshape.view(-1)
                        else:
                            update = update_reshape
                    # ==============================
                    # Last Dim is R, First Dim is S{} means split dim 0  --->
                    # Row Parallel ---> sq_col need Do (row) Reduce
                    # ==============================
                    elif self.shard_spec.sharding_sequence[-1] == "R":
                        # Row Residual situation
                        if self.grad_shape[0] % self.data_parallel_size != 0:
                            # gather update[flatten] along dp group then reshape to [H/tp, W]
                            update = _gather(
                                input_=update, dim=-1, process_group=self.data_parallel_group
                            )
                            # view update to origin[tp] shape
                            update_reshape = update.view(-1, self.grad_shape[1])
                            # gather grad[flatten] along dp group then reshape to [H/tp, W]
                            grad = _gather(
                                input_=grad, dim=-1, process_group=self.data_parallel_group
                            ) 
                            grad_reshape = grad.view(-1, self.grad_shape[1])
                            exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/tp]
                            exp_avg_sq_col = state["exp_avg_sq_col"]  # [W]
                            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                            # reduce col
                            dist.all_reduce(exp_avg_sq_col, group=self.tensor_parallel_group)
                            exp_avg_sq_col.div_(self.tensor_parallel_size)
                            update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                            update_reshape.mul_(grad_reshape)
                            if self.use_zero:
                                update = _split(input_=update_reshape.view(-1), dim=-1, process_group=self.data_parallel_group)
                            else:
                                update = update_reshape
                        else:
                            update_reshape = update.view(-1, self.grad_shape[1])
                            grad_reshape = grad.view(-1, self.grad_shape[1])
                            exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/dp/tp]
                            exp_avg_sq_col = state["exp_avg_sq_col"]  # [W]
                            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                            # reduce col
                            dist.all_reduce(exp_avg_sq_col, group=self.tensor_parallel_group)
                            exp_avg_sq_col.div_(self.tensor_parallel_size)
                            # gather row
                            exp_avg_sq_row_gather = _gather(
                                input_=exp_avg_sq_row, dim=-1, process_group=self.tensor_parallel_group
                            )
                            sq_row_meam = exp_avg_sq_row_gather.mean(dim=-1, keepdim=True)
                            update_reshape = self._approx_sq_grad_row_parallel(exp_avg_sq_row, exp_avg_sq_col, sq_row_meam)
                            update_reshape.mul_(grad_reshape)
                            if self.use_zero:
                                update = update_reshape.view(-1)
                            else:
                                update = update_reshape
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                # (Line No.8) RMS
                # update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)
                if self.use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.add_(p, alpha=(-group["weight_decay"] * lr))
                    
                p.add_(-update)


        return loss

import math
from typing import Dict

import torch
import torch.distributed as dist

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
        lr = None
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
        self.tp_size = 1
        self.tp_group = None
        self.dp_size = 1
        self.dp_group = None
        self.shard_to_working_param = None  # Dict{id:shape}, sample {id(param): torch.tensor}
        self.use_zero = True

        self.param_is_dtensor_dict = {}  # {id(p): True/False}
        self.grad_shape_dict = {}  # {id(p): master param shape}
        self.factored_dict = {}  # {id(p): True/False}
        self.use_first_moment_dict = {}  # {id(p): True/False}
        self.shard_spec_dict = {}  # {id(p): ShardSpec}
        super().__init__(params, defaults)

    def setup_distributed(
        self,
        tp_group: dist.ProcessGroup = None,
        dp_group: dist.ProcessGroup = None,
        shard_to_working_param: Dict = {},
        padding_map=None,
        use_zero: bool = True,
    ) -> None:
        """Setup process groups for TP and ZeRO 2.
        Inject features to the Optimizer

        Args:
            tp_group: The devices group for tensor parallel;
            dp_group: The devices group for data parallel;
            shard_to_working_param (Dict): ZeRO 2 feeds the optimizer a sharded param view as grads are sharded.
                This maps from id(view) to working params used in forward & backward.
            padding_map: An empty interface placeholder;
            use_zero: Whether or not to use zero;

        """
        self.tp_group = tp_group  # "Expected row process group"
        self.dp_group = dp_group
        if self.tp_group is not None:
            self.tp_size = dist.get_world_size(self.tp_group)
        if self.dp_group is not None:
            self.dp_size = dist.get_world_size(self.dp_group)
        self.use_zero = use_zero

        self.shard_to_working_param = shard_to_working_param if shard_to_working_param is not None else {}
        # grad is None, cause we dont setup now
        for group in self.param_groups:
            for p in group["params"]:
                self.shard_to_working_param[id(p)] = self.shard_to_working_param.get(
                    id(p), p
                )  # If not ZeRO, working param is master param
                self.param_is_dtensor_dict[id(p)] = is_distributed_tensor(self.shard_to_working_param[id(p)])
                self.grad_shape_dict[id(p)] = self.shard_to_working_param.get(id(p)).shape
                self.factored_dict[id(p)], self.use_first_moment_dict[id(p)] = self._get_options(
                    group, self.grad_shape_dict[id(p)]
                )
                if self.param_is_dtensor_dict[id(p)]:
                    self.shard_spec_dict[id(p)] = get_sharding_spec(self.shard_to_working_param[id(p)])
                else:
                    self.shard_spec_dict[id(p)] = None

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
    def _rms(tensor, param_is_dtensor, use_zero, tp_size, dp_size, tp_group, dp_group):
        tensor_sum = tensor.pow(2).sum()
        num_of_element = tensor.numel()

        if param_is_dtensor:
            # reduce tensor_sum  from tp_group
            dist.all_reduce(tensor_sum, group=tp_group)
            num_of_element = num_of_element * tp_size
        if use_zero:
            dist.all_reduce(tensor_sum, group=dp_group)
            num_of_element = num_of_element * dp_size
        rms = (tensor_sum / num_of_element).sqrt()
        return rms

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

    def _col_parallel_factor(self, update, grad, state, grad_shape, beta2t):
        if grad_shape[0] % self.dp_size != 0:
            # gather update[flatten] along dp group then reshape to [H, W/tp]
            update = _gather(input_=update, dim=-1, process_group=self.dp_group)
            update_reshape = update.view(-1, grad_shape[1])
            # gather grad[flatten] along dp group then reshape to [H, W/tp]
            grad = _gather(input_=grad, dim=-1, process_group=self.dp_group)
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state["exp_avg_sq_row"]  # [H]
            exp_avg_sq_col = state["exp_avg_sq_col"]  # [W/tp]
            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
            update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update_reshape.mul_(grad_reshape)
        else:
            update_reshape = update.view(-1, grad_shape[1])
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/dp]
            exp_avg_sq_col = state["exp_avg_sq_col"]  # [W/tp]
            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
            dist.all_reduce(exp_avg_sq_row, group=self.tp_group)
            exp_avg_sq_row.div_(self.tp_size)
            update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update_reshape.mul_(grad_reshape)

        if self.use_zero:
            update = update_reshape.view(-1)
        else:
            update = update_reshape
        return update

    def _row_parallel_factor(self, update, grad, state, grad_shape, beta2t):
        if grad_shape[0] % self.dp_size != 0:
            # gather update[flatten] along dp group then reshape to [H/tp, W]
            update = _gather(input_=update, dim=-1, process_group=self.dp_group)
            # view update to origin[tp] shape
            update_reshape = update.view(-1, grad_shape[1])
            # gather grad[flatten] along dp group then reshape to [H/tp, W]
            grad = _gather(input_=grad, dim=-1, process_group=self.dp_group)
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/tp]
            exp_avg_sq_col = state["exp_avg_sq_col"]  # [W]
            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
            # reduce col
            dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
            exp_avg_sq_col.div_(self.tp_size)
            update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update_reshape.mul_(grad_reshape)
            if self.use_zero:
                update = _split(input_=update_reshape.view(-1), dim=-1, process_group=self.dp_group)
            else:
                update = update_reshape
        else:
            update_reshape = update.view(-1, grad_shape[1])
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/dp/tp]
            exp_avg_sq_col = state["exp_avg_sq_col"]  # [W]
            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
            # reduce col
            dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
            exp_avg_sq_col.div_(self.tp_size)
            # gather row
            exp_avg_sq_row_gather = _gather(input_=exp_avg_sq_row, dim=-1, process_group=self.tp_group)
            sq_row_meam = exp_avg_sq_row_gather.mean(dim=-1, keepdim=True)
            update_reshape = self._approx_sq_grad_row_parallel(exp_avg_sq_row, exp_avg_sq_col, sq_row_meam)
            update_reshape.mul_(grad_reshape)
            if self.use_zero:
                update = update_reshape.view(-1)
            else:
                update = update_reshape
        return update

    def _base_factor(self, update, grad, state, grad_shape, beta2t):
        if self.use_zero:
            # only zero
            if grad_shape[0] % self.dp_size != 0:
                # view update to origin shape update.view(grad_shape[0]//self.data_parallel_size , grad_shape[1])
                # row mean no change
                # col mean need reduce and div
                # gather update[flatten] along dp group then reshape to [H, W]
                update = _gather(input_=update, dim=-1, process_group=self.dp_group)
                # view update to origin[tp] shape
                update_reshape = update.view(-1, grad_shape[1])
                # gather grad[flatten] along dp group then reshape to [H, W]
                grad = _gather(input_=grad, dim=-1, process_group=self.dp_group)
                grad_reshape = grad.view(-1, grad_shape[1])
                exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/dp]
                exp_avg_sq_col = state["exp_avg_sq_col"]  # [W]
                exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                # reduce col
                dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
                exp_avg_sq_col.div_(self.tp_size)
                update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                update_reshape.mul_(grad_reshape)
                update = _split(input_=update_reshape.view(-1), dim=-1, process_group=self.dp_group)
            else:
                # no residual row
                # view update to origin[tp] shape
                update_reshape = update.view(-1, grad_shape[1])  # [H/dp, W]
                grad_reshape = grad.view(-1, grad_shape[1])  # [H/dp, W]
                exp_avg_sq_row = state["exp_avg_sq_row"]  # [H/tp]
                exp_avg_sq_col = state["exp_avg_sq_col"]  # [W]
                exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                # reduce col
                dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
                exp_avg_sq_col.div_(self.tp_size)
                update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                update_reshape.mul_(grad_reshape)
                update = update_reshape.view(-1)
        else:
            # base factor; no tp, no dp
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            # Exponential average of row indexes
            exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
            # Exponential average of columns indexes
            exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
            # Approximation of exponential moving average of square of gradient
            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        return update

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
                grad_shape = self.grad_shape_dict[id(p)]
                param_is_dtensor = self.param_is_dtensor_dict[id(p)]
                if param_is_dtensor:
                    grad_shape = self.shard_to_working_param.get(id(p)).shape  # tp shape (2 dim)
                factored, use_first_moment = self.factored_dict[id(p)], self.use_first_moment_dict[id(p)]

                shard_spec = self.shard_spec_dict[id(p)]
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)
                    if factored:
                        if param_is_dtensor:
                            if shard_spec.sharding_sequence[0] == "R":  # Col Parallel
                                if grad_shape[0] % self.dp_size != 0:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0], device=p.device, dtype=p.dtype
                                    )  # [H]
                                else:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=p.device, dtype=p.dtype
                                    )  # [H/dp]
                                state["exp_avg_sq_col"] = torch.zeros(
                                    grad_shape[1], device=p.device, dtype=p.dtype
                                )  # [W/TP]

                            if shard_spec.sharding_sequence[-1] == "R":  # Row Parallel
                                # Row indivisible shape situation
                                if grad_shape[0] % self.dp_size != 0:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0], device=p.device, dtype=p.dtype
                                    )  # [H/tp]
                                else:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=p.device, dtype=p.dtype
                                    )  # [H/dp/tp]

                                state["exp_avg_sq_col"] = torch.zeros(
                                    grad_shape[1], device=p.device, dtype=p.dtype
                                )  # [W]
                        else:
                            if self.use_zero:
                                if grad_shape[0] % self.dp_size != 0:
                                    # save all exp_avg_sq_row [H]
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0], device=grad.device, dtype=p.dtype
                                    )
                                else:
                                    # exp_avg_sq_row [H // dp]
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=grad.device, dtype=p.dtype
                                    )
                            else:
                                # exp_avg_sq_row [H]
                                state["exp_avg_sq_row"] = torch.zeros(grad_shape[0], device=grad.device, dtype=p.dtype)
                            # exp_avg_sq_col alaways [W]
                            state["exp_avg_sq_col"] = torch.zeros(grad_shape[1], device=grad.device, dtype=p.dtype)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"]
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"]
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"]
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"]

                state["step"] += 1
                lr = self._get_lr(group, state)
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]

                if factored:
                    if param_is_dtensor:
                        # ==============================
                        # First Dim is R, Last Dim is S{} means split dim -1  --->
                        # Coloum Parallel ---> sq_row need Do (col) Reduce
                        # ==============================
                        if shard_spec.sharding_sequence[0] == "R":
                            update = self._col_parallel_factor(update, grad, state, grad_shape, beta2t)
                        # ==============================
                        # Last Dim is R, First Dim is S{} means split dim 0  --->
                        # Row Parallel ---> sq_col need Do (row) Reduce
                        # ==============================
                        elif shard_spec.sharding_sequence[-1] == "R":
                            update = self._row_parallel_factor(update, grad, state, grad_shape, beta2t)
                    else:
                        update = self._base_factor(update, grad, state, grad_shape, beta2t)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                # # (Line No.8) RMS
                rms = self._rms(
                    update,
                    param_is_dtensor,
                    self.use_zero,
                    self.tp_size,
                    self.dp_size,
                    self.tp_group,
                    self.dp_group,
                )
                update.div_((rms / group["clip_threshold"]).clamp_(min=1.0))

                update.mul_(lr)
                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.add_(p, alpha=(-group["weight_decay"] * lr))

                p.add_(-update)

        return loss

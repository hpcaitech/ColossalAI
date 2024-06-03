from typing import Dict

import torch
import torch.distributed as dist

from colossalai.interface.optimizer import DistributedOptim
from colossalai.shardformer.layer._operation import _gather, _split
from colossalai.tensor.d_tensor import get_sharding_spec, is_distributed_tensor


class DistributedCAME(DistributedOptim):
    """Implements CAME algorithm.
    This implementation is based on:
    `CAME: Confidence-guided Adaptive Memory Efficient Optimization`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of
            final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
        update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )

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

        super(DistributedCAME, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def setup_distributed(
        self,
        tp_group: dist.ProcessGroup = None,
        dp_group: dist.ProcessGroup = None,
        shard_to_working_param: Dict = {},
        padding_map=None,
        use_zero: bool = True,
    ) -> None:
        """
        Inject features to the Optimizer

        Args:
            tp_group: The devices group for tensor parallel;
            dp_group: The devices group for data parallel;
            shard_to_working_param (Dict): ZeRO 2 feeds the optimizer a sharded param view as grads are sharded.
                This maps from id(view) to working params used in forward & backward.
            padding_map: Interface placeholder
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
                # w/o ZeRO: master param = working param
                self.shard_to_working_param[id(p)] = self.shard_to_working_param.get(id(p), p)
                self.param_is_dtensor_dict[id(p)] = is_distributed_tensor(self.shard_to_working_param[id(p)])
                self.grad_shape_dict[id(p)] = self.shard_to_working_param[id(p)].shape
                # Avoid row parallel lead H=1, then factored param is determined as not factored;
                if self.param_is_dtensor_dict[id(p)]:
                    self.shard_spec_dict[id(p)] = get_sharding_spec(self.shard_to_working_param[id(p)])
                    if self.shard_spec_dict[id(p)].sharding_sequence[0] == "R":
                        self.factored_dict[id(p)] = True
                    elif self.shard_spec_dict[id(p)].sharding_sequence[-1] == "R":
                        self.factored_dict[id(p)] = True
                    else:
                        self.factored_dict[id(p)] = self._get_options(self.grad_shape_dict[id(p)])

                else:
                    self.shard_spec_dict[id(p)] = None
                    self.factored_dict[id(p)] = self._get_options(self.grad_shape_dict[id(p)])

    @staticmethod
    def _get_options(param_shape):
        factored = len(param_shape) >= 2
        return factored

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
        r_factor = (exp_avg_sq_row / sq_row_meam).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _col_parallel_factor(self, update, grad, state_row, state_col, grad_shape, beta2t):
        if grad_shape[0] % self.dp_size != 0:
            # gather update[flatten] along dp group then reshape to [H, W/tp]
            update = _gather(input_=update, dim=-1, process_group=self.dp_group)
            update_reshape = update.view(-1, grad_shape[1])
            # gather grad[flatten] along dp group then reshape to [H, W/tp]
            grad = _gather(input_=grad, dim=-1, process_group=self.dp_group)
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state_row  # [H]
            exp_avg_sq_col = state_col  # [W/tp]
            exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
            exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
            update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update_reshape.mul_(grad_reshape)
        else:
            update_reshape = update.view(-1, grad_shape[1])
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state_row  # [H]
            exp_avg_sq_col = state_col  # [W/tp]
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

    def _row_parallel_factor(self, update, grad, state_row, state_col, grad_shape, beta2t):
        if grad_shape[0] % self.dp_size != 0:
            # gather update[flatten] along dp group then reshape to [H/tp, W]
            update = _gather(input_=update, dim=-1, process_group=self.dp_group)
            # view update to origin[tp] shape
            update_reshape = update.view(-1, grad_shape[1])
            # gather grad[flatten] along dp group then reshape to [H/tp, W]
            grad = _gather(input_=grad, dim=-1, process_group=self.dp_group)
            grad_reshape = grad.view(-1, grad_shape[1])
            exp_avg_sq_row = state_row  # [H]
            exp_avg_sq_col = state_col  # [W/tp]
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
            exp_avg_sq_row = state_row  # [H]
            exp_avg_sq_col = state_col  # [W/tp]
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

    def _base_factor(self, update, grad, state_row, state_col, grad_shape, beta2t):
        if self.use_zero:
            # only zero
            #  [30522, 128], [2, 128]
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
                exp_avg_sq_row = state_row  # [H/dp]
                exp_avg_sq_col = state_col  # [W]
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
                exp_avg_sq_row = state_row  # [H/dp]
                exp_avg_sq_col = state_col  # [W]
                exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                # reduce col
                dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
                exp_avg_sq_col.div_(self.tp_size)
                update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                update_reshape.mul_(grad_reshape)
                update = update_reshape.view(-1)
        else:
            # # base factor; no tp, no dp
            exp_avg_sq_row = state_row  # [H/dp]
            exp_avg_sq_col = state_col  # [W]
            # Exponential average of row indexes
            exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
            # Exponential average of columns indexes
            exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
            # Approximation of exponential moving average of square of gradient
            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        return update

    # factor
    def _base_res_factor(self, res, exp_avg, state_row, state_col, grad_shape, beta2t):
        if self.use_zero:
            # only zero
            if grad_shape[0] % self.dp_size != 0:
                # view res to origin shape res.view(grad_shape[0]//self.data_parallel_size , grad_shape[1])
                # row mean no change
                # col mean need reduce and div
                # gather res[flatten] along dp group then reshape to [H, W]
                res = _gather(input_=res, dim=-1, process_group=self.dp_group)
                # view res to origin[tp] shape
                res_reshape = res.view(-1, grad_shape[1])
                # gather exp_avg[flatten] along dp group then reshape to [H, W]
                exp_avg = _gather(input_=exp_avg, dim=-1, process_group=self.dp_group)
                exp_avg_reshape = exp_avg.view(-1, grad_shape[1])
                exp_avg_sq_row = state_row  # [H/dp]
                exp_avg_sq_col = state_col  # [W]
                exp_avg_sq_row.mul_(beta2t).add_(res_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                exp_avg_sq_col.mul_(beta2t).add_(res_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                # reduce col
                dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
                exp_avg_sq_col.div_(self.tp_size)
                res_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                res_reshape.mul_(exp_avg_reshape)
                res = _split(input_=res_reshape.view(-1), dim=-1, process_group=self.dp_group)
            else:
                # no residual row
                # view res to origin[tp] shape
                res_reshape = res.view(-1, grad_shape[1])  # [H/dp, W]
                exp_avg_reshape = exp_avg.view(-1, grad_shape[1])  # [H/dp, W]
                exp_avg_sq_row = state_row  # [H/dp]
                exp_avg_sq_col = state_col  # [W]
                exp_avg_sq_row.mul_(beta2t).add_(res_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                exp_avg_sq_col.mul_(beta2t).add_(res_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                # reduce col
                dist.all_reduce(exp_avg_sq_col, group=self.tp_group)
                exp_avg_sq_col.div_(self.tp_size)
                res_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                res_reshape.mul_(exp_avg_reshape)
                res = res_reshape.view(-1)
        else:
            # # base factor; no tp, no dp
            exp_avg_sq_row = state_row  # [H/dp]
            exp_avg_sq_col = state_col  # [W]
            # Exponential average of row indexes
            exp_avg_sq_row.mul_(beta2t).add_(res.mean(dim=-1), alpha=(1.0 - beta2t))
            # Exponential average of columns indexes
            exp_avg_sq_col.mul_(beta2t).add_(res.mean(dim=-2), alpha=(1.0 - beta2t))
            # Approximation of exponential moving average of square of gradient
            res = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            res.mul_(exp_avg)
        return res

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
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
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                # Under zero the grad_shape is the original grad that is flattened and then cut (only one dimension)
                grad_shape = grad.shape
                grad_shape = self.grad_shape_dict[id(p)]
                param_is_dtensor = self.param_is_dtensor_dict[id(p)]
                if param_is_dtensor:
                    grad_shape = self.shard_to_working_param.get(id(p)).shape  # tp shape (2 dim)
                factored = self.factored_dict[id(p)]
                shard_spec = self.shard_spec_dict[id(p)]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    if factored:
                        if param_is_dtensor:
                            if shard_spec.sharding_sequence[0] == "R":  # Col Parallel
                                if grad_shape[0] % self.dp_size != 0:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0], device=p.device, dtype=p.dtype
                                    )  # [H]
                                    state["exp_avg_res_row"] = torch.zeros(
                                        grad_shape[0], device=p.device, dtype=p.dtype
                                    )  # [H]
                                else:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=p.device, dtype=p.dtype
                                    )  # [H/dp]
                                    state["exp_avg_res_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=p.device, dtype=p.dtype
                                    )  # [H/dp]
                                state["exp_avg_sq_col"] = torch.zeros(
                                    grad_shape[1], device=p.device, dtype=p.dtype
                                )  # [W/TP]
                                state["exp_avg_res_col"] = torch.zeros(
                                    grad_shape[1], device=p.device, dtype=p.dtype
                                )  # [W/TP]

                            if shard_spec.sharding_sequence[-1] == "R":  # Row Parallel
                                # Row indivisible shape situation
                                if grad_shape[0] % self.dp_size != 0:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0], device=p.device, dtype=p.dtype
                                    )  # [H/tp]
                                    state["exp_avg_res_row"] = torch.zeros(
                                        grad_shape[0], device=p.device, dtype=p.dtype
                                    )  # [H/tp]
                                else:
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=p.device, dtype=p.dtype
                                    )  # [H/dp/tp]
                                    state["exp_avg_res_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=p.device, dtype=p.dtype
                                    )  # [H/dp/tp]

                                state["exp_avg_sq_col"] = torch.zeros(
                                    grad_shape[1], device=p.device, dtype=p.dtype
                                )  # [W]
                                state["exp_avg_res_col"] = torch.zeros(
                                    grad_shape[1], device=p.device, dtype=p.dtype
                                )  # [W]
                        else:
                            if self.use_zero:
                                if grad_shape[0] % self.dp_size != 0:
                                    # save all exp_avg_sq_row [H]
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0], device=grad.device, dtype=p.dtype
                                    )
                                    state["exp_avg_res_row"] = torch.zeros(
                                        grad_shape[0], device=grad.device, dtype=p.dtype
                                    )
                                else:
                                    # exp_avg_sq_row [H // dp]
                                    state["exp_avg_sq_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=grad.device, dtype=p.dtype
                                    )
                                    state["exp_avg_res_row"] = torch.zeros(
                                        grad_shape[0] // self.dp_size, device=grad.device, dtype=p.dtype
                                    )
                            else:
                                # exp_avg_sq_row [H]
                                state["exp_avg_sq_row"] = torch.zeros(grad_shape[0], device=grad.device, dtype=p.dtype)
                                state["exp_avg_res_row"] = torch.zeros(grad_shape[0], device=grad.device, dtype=p.dtype)
                            # exp_avg_sq_col alaways [W]
                            state["exp_avg_sq_col"] = torch.zeros(grad_shape[1], device=grad.device, dtype=p.dtype)
                            state["exp_avg_res_col"] = torch.zeros(grad_shape[1], device=grad.device, dtype=p.dtype)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["RMS"] = 0
                else:
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"]
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"]
                        state["exp_avg_res_row"] = state["exp_avg_sq_row"]
                        state["exp_avg_res_col"] = state["exp_avg_sq_col"]
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"]

                state["step"] += 1

                update = (grad**2) + group["eps"][0]
                if factored:
                    if param_is_dtensor:
                        # ==============================
                        # First Dim is R, Last Dim is S{} means split dim -1  --->
                        # Coloum Parallel ---> sq_row need Do (col) Reduce
                        # ==============================
                        if shard_spec.sharding_sequence[0] == "R":
                            update = self._col_parallel_factor(
                                update,
                                grad,
                                state["exp_avg_sq_row"],
                                state["exp_avg_sq_col"],
                                grad_shape,
                                group["betas"][1],
                            )
                        # ==============================
                        # Last Dim is R, First Dim is S{} means split dim 0  --->
                        # Row Parallel ---> sq_col need Do (row) Reduce
                        # ==============================
                        elif shard_spec.sharding_sequence[-1] == "R":
                            update = self._row_parallel_factor(
                                update,
                                grad,
                                state["exp_avg_sq_row"],
                                state["exp_avg_sq_col"],
                                grad_shape,
                                group["betas"][1],
                            )
                    else:
                        update = self._base_factor(
                            update,
                            grad,
                            state["exp_avg_sq_row"],
                            state["exp_avg_sq_col"],
                            grad_shape,
                            group["betas"][1],
                        )
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=(1.0 - group["betas"][1]))
                    update = exp_avg_sq.rsqrt().mul_(grad)
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

                exp_avg = state["exp_avg"]
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])
                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg) ** 2 + group["eps"][1]
                if factored:
                    if param_is_dtensor:
                        # ==============================
                        # First Dim is R, Last Dim is S{} means split dim -1  --->
                        # Coloum Parallel ---> sq_row need Do (col) Reduce
                        # ==============================
                        if shard_spec.sharding_sequence[0] == "R":
                            update = self._col_parallel_factor(
                                res,
                                exp_avg,
                                state["exp_avg_res_row"],
                                state["exp_avg_res_col"],
                                grad_shape,
                                group["betas"][2],
                            )
                        # ==============================
                        # Last Dim is R, First Dim is S{} means split dim 0  --->
                        # Row Parallel ---> sq_col need Do (row) Reduce
                        # ==============================
                        elif shard_spec.sharding_sequence[-1] == "R":
                            update = self._row_parallel_factor(
                                res,
                                exp_avg,
                                state["exp_avg_res_row"],
                                state["exp_avg_res_col"],
                                grad_shape,
                                group["betas"][2],
                            )
                    else:
                        update = self._base_res_factor(
                            res,
                            exp_avg,
                            state["exp_avg_res_row"],
                            state["exp_avg_res_col"],
                            grad_shape,
                            group["betas"][2],
                        )
                else:
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["weight_decay"] * group["lr"])
                update.mul_(group["lr"])
                p.add_(-update)
        return loss

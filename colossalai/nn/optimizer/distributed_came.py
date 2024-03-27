import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.interface.optimizer import DistributedOptimizer
from colossalai.tensor.d_tensor import api


class DistributedCAME(DistributedOptimizer):
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
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super(DistributedCAME, self).__init__(params, defaults)

        self.distributed = False
        self.zero = False
        self.clip_method = dict()
        self.ori_shape = dict()
        self.working_shape = dict()
        self.gather_before_compute = dict()
        # record working parameter original shape (Before TP)
        for group in self.param_groups:
            for p in group["params"]:
                self.ori_shape[id(p)] = p.data.shape

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def setup_distributed(
        self, master_to_working_map: dict, tp_group: ProcessGroup, zero_group: ProcessGroup, zero_flag: bool = False
    ):
        self.tensor_parallel_group = tp_group
        self.zero_parallel_group = zero_group
        # When running setup_distribute(), the parameters now are master parameters (After TP and zero)
        # But master param != d_tensor, need to record information of working param (After TP before zero)
        for group in self.param_groups:
            for p in group["params"]:
                # if no zero, working param == master param
                working_param = master_to_working_map[id(p)] if master_to_working_map else p
                self.ori_shape[id(p)] = self.ori_shape[id(working_param)]
                self.working_shape[id(p)] = working_param.data.shape
                self.gather_before_compute[id(p)] = False
                try:
                    api.get_device_mesh(working_param)
                    sharding_spec = api.get_sharding_spec(working_param)
                    self.clip_method[id(p)] = "col" if 0 in sharding_spec.dim_partition_dict.keys() else "row"
                    # if dist.get_rank() == 0:
                    #     print(self.ori_shape[id(p)], p.data.shape, sharding_spec.dim_partition_dict, self.clip_method[id(p)])
                except:
                    self.clip_method[id(p)] = None

        self.zero = True if master_to_working_map and zero_flag else False
        self.distributed = True
        # print(self.distributed, self.zero, dist.get_world_size(zero_group))

    def _get_options(self, param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor, param):
        if not self.distributed:
            return tensor.norm(2) / (tensor.numel() ** 0.5)
        # return tensor.norm(2) / (tensor.numel() ** 0.5)
        # 计算当前设备上张量的平方和
        local_sum_sq = tensor.pow(2).sum()

        # 在所有设备上汇总平方和
        global_sum_sq = local_sum_sq.clone()
        dist.all_reduce(global_sum_sq, op=dist.ReduceOp.SUM, group=self.tensor_parallel_group)

        # 在所有设备上汇总元素总数
        local_numel = torch.tensor(tensor.numel(), device=tensor.device)
        global_numel = local_numel.clone()
        dist.all_reduce(global_numel, op=dist.ReduceOp.SUM, group=self.tensor_parallel_group)
        if self.zero and not self.gather_before_compute[id(param)]:
            dist.all_reduce(global_sum_sq, op=dist.ReduceOp.SUM, group=self.zero_parallel_group)
            dist.all_reduce(global_numel, op=dist.ReduceOp.SUM, group=self.zero_parallel_group)

        # 计算 RMS
        rms = (global_sum_sq / global_numel).sqrt()
        return rms

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, param):
        exp_avg_sq_row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True)
        if self.distributed:
            clip_method = self.clip_method[id(param)]
            if clip_method == "col":
                dist.all_reduce(exp_avg_sq_row_mean, op=dist.ReduceOp.AVG, group=self.tensor_parallel_group)
            if self.zero and not self.gather_before_compute[id(param)]:
                dist.all_reduce(exp_avg_sq_row_mean, op=dist.ReduceOp.AVG, group=self.zero_parallel_group)
        r_factor = (exp_avg_sq_row / exp_avg_sq_row_mean).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()

        return torch.mul(r_factor, c_factor)

    def _unflatten_grad_tensor_by_param(self, param):
        ori_shape = self.ori_shape[id(param)]
        self.working_shape[id(param)]
        # return param.grad.data.reshape(*working_shape)
        if not (len(ori_shape) >= 2 and len(param.grad.data.shape) == 1):
            return param.grad.data
        remaining_dims = ori_shape[1:]
        if param.grad.data.shape[0] % torch.prod(torch.tensor(remaining_dims)) != 0:
            self.gather_before_compute[id(param)] = True
            gathered_grad = [
                torch.zeros_like(param.grad.data) for _ in range(dist.get_world_size(group=self.zero_parallel_group))
            ]
            dist.all_gather(gathered_grad, param.grad.data, group=self.zero_parallel_group)
            gathered_grad = torch.cat(gathered_grad, dim=-1).reshape(*ori_shape)
            assert gathered_grad.shape == ori_shape
            return gathered_grad
        return param.grad.data.reshape(-1, *remaining_dims)

    def _flatten_update_tensor_by_param(self, param, tensor):
        self.working_shape[id(param)]
        # return tensor.reshape(*working_shape)
        ori_shape = self.ori_shape[id(param)]
        if not (len(ori_shape) >= 2 and len(param.grad.data.shape) == 1):
            return tensor
        if self.gather_before_compute[id(param)]:
            rank = dist.get_rank(group=self.zero_parallel_group)
            length = param.data.shape[0]
            return torch.flatten(tensor)[rank * length : (rank + 1) * length]
        return torch.flatten(tensor)

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
                grad = self._unflatten_grad_tensor_by_param(p) if self.zero else p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                # zero下grad_shape是原始grad经过flatten后再切割(只有一维)
                grad_shape = grad.shape

                factored = self._get_options(grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)

                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data, p)

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    # 局部平均
                    sq_mean_row = update.mean(dim=-1)
                    sq_mean_col = update.mean(dim=-2)
                    if self.distributed:
                        # 全局同步
                        if self.clip_method[id(p)] == "row":
                            dist.all_reduce(sq_mean_row, op=dist.ReduceOp.AVG, group=self.tensor_parallel_group)
                        elif self.clip_method[id(p)] == "col":
                            dist.all_reduce(sq_mean_col, op=dist.ReduceOp.AVG, group=self.tensor_parallel_group)
                        else:
                            pass
                        if self.zero and not self.gather_before_compute[id(p)]:
                            dist.all_reduce(sq_mean_col, op=dist.ReduceOp.AVG, group=self.zero_parallel_group)

                    # 得到的exp_avg是完整exp_avg的切割
                    exp_avg_sq_row.mul_(group["betas"][1]).add_(sq_mean_row, alpha=1.0 - group["betas"][1])
                    exp_avg_sq_col.mul_(group["betas"][1]).add_(sq_mean_col, alpha=1.0 - group["betas"][1])

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, p)
                    update.mul_(grad)
                else:
                    # bias执行这个部分
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    update = exp_avg_sq.rsqrt().mul_(grad)
                # shard_grad_list = [torch.zeros_like(grad).to("cuda") for _ in range(dist.get_world_size(self.tensor_parallel_group))]
                # dist.all_gather(shard_grad_list, grad, self.tensor_parallel_group)
                # shard_grad = torch.cat(shard_grad_list, dim=1)
                # if dist.get_rank() == 0:
                #     print("dist: ", torch.sum(shard_grad), shard_grad)

                # update也为完整update的切割
                update.div_((self._rms(update, p) / group["clip_threshold"]).clamp_(min=1.0))
                exp_avg = state["exp_avg"]
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg) ** 2 + group["eps"][0]
                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    res_mean_row = res.mean(dim=-1)
                    res_mean_col = res.mean(dim=-2)
                    if self.distributed:
                        if self.clip_method[id(p)] == "row":
                            dist.all_reduce(res_mean_row, op=dist.ReduceOp.AVG, group=self.tensor_parallel_group)
                        elif self.clip_method[id(p)] == "col":
                            dist.all_reduce(res_mean_col, op=dist.ReduceOp.AVG, group=self.tensor_parallel_group)
                        else:
                            pass
                        if self.zero and not self.gather_before_compute[id(p)]:
                            dist.all_reduce(res_mean_col, op=dist.ReduceOp.AVG, group=self.zero_parallel_group)

                    exp_avg_res_row.mul_(group["betas"][2]).add_(res_mean_row, alpha=1.0 - group["betas"][2])
                    exp_avg_res_col.mul_(group["betas"][2]).add_(res_mean_col, alpha=1.0 - group["betas"][2])

                    # Approximation of exponential moving average of instability
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col, p)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"]["lr"])
                update.mul_(group["lr"]["lr"])
                if self.zero:
                    update = self._flatten_update_tensor_by_param(p, update)
                p.data.add_(-update)

        return loss

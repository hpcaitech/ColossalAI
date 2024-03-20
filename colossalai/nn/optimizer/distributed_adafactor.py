import math
import os 
import torch
from torch.optim import Optimizer
import torch.distributed as dist

def _gather(input_: torch.Tensor, group_:torch.distributed.ProcessGroup) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = group_

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


# DistributedAdaFactor (with Tensor parallel and Zero stage 2)
class DistributedAdaFactor(Optimizer):
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
        super().__init__(params, defaults)
        # self.tensor_parallel_size = device_mesh._physical_mesh_id.shape[0]
        # self.tensor_parallel_group = device_mesh.get_process_group(axis=1) # "Expected row process group"
        # self.localRank = int(os.environ['LOCAL_RANK']) 
        # self.worldSize = int(os.environ['WORLD_SIZE']) 
        # self.sharding_spec_dict = sharding_spec_dict
        # self.param_shape = param_shape # Dict{id:shape}, sample {id(weight): torch.Size(4,4)}
        
        self.localRank = None
        self.worldSize = None
        self.tensor_parallel_size = None
        self.tensor_parallel_group = None
        self.sharding_spec_dict = None
        self.param_shape = None # Dict{id:shape}, sample {id(weight): torch.Size(4,4)}
        # self.setup_distribute(defaults)
    
    def setup_distribute(self, device_mesh, sharding_spec_dict, param_shape):
        # device_mesh = defaults['device_mesh']
        # sharding_spec_dict = defaults['sharding_spec_dict']
        # self.tensor_parallel_size = device_mesh._physical_mesh_id.shape[0]
        # self.tensor_parallel_group = device_mesh.get_process_group(axis=1) # "Expected row process group"
        # self.localRank = int(os.environ['LOCAL_RANK']) 
        # self.worldSize = int(os.environ['WORLD_SIZE']) 
        # self.sharding_spec_dict = sharding_spec_dict
        # self.param_shape = defaults['param_shape'] # Dict{id:shape}, sample {id(weight): torch.Size(4,4)}
        device_mesh = device_mesh
        self.tensor_parallel_size = device_mesh._physical_mesh_id.shape[0]
        self.tensor_parallel_group = device_mesh.get_process_group(axis=1) # "Expected row process group"
        self.localRank = int(os.environ['LOCAL_RANK']) 
        self.worldSize = int(os.environ['WORLD_SIZE']) 
        self.sharding_spec_dict = sharding_spec_dict
        self.param_shape = param_shape # Dict{id:shape}, sample {id(weight): torch.Size(4,4)}

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
        factored = (len(param_shape) >= 2) 
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
        # print(f"param_groups {self.param_groups}")
        for group in self.param_groups:
            # update weight & bias
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # ==============================
                # Init Current Param Shape
                # ==============================
                p_shape = None  # param shape (not split)
                param_height = None
                param_width = None
                param_height_parallel = None
                param_width_parallel = None
                sharding_spec = None
                if id(p) in self.param_shape.keys():
                    p_shape = self.param_shape[id(p)]
                    # if id(p) in self.sharding_spec_dict.keys() and self.sharding_spec_dict[id(p)] != None:
                    #     sharding_spec = self.sharding_spec_dict[id(p)]
                    if len(p_shape) >= 2: # factored
                        param_height = p_shape[0]
                        param_width = p_shape[1]
                        sharding_spec = self.sharding_spec_dict[id(p)]
                        if sharding_spec.sharding_sequence[0] == 'R': # Col Parallel 
                            param_height_parallel = param_height
                            param_width_parallel = param_width // self.tensor_parallel_size # W/N
                        if sharding_spec.sharding_sequence[-1] == 'R': # Row Parallel
                            param_height_parallel = param_height // self.tensor_parallel_size # H/N
                            param_width_parallel = param_width
                
                # grad shape is same as weigh / bias
                grad = p.grad.to(self.localRank)
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                
                """
                p (group["params"]) is weight
                state 
                {'step', 
                'exp_avg_sq_row', 
                'exp_avg_sq_col', 
                'RMS'
                }
                
                p (group["params"]) is bias
                state 
                {'step', 
                'exp_avg_sq', 
                'RMS'
                }
                """
                state = self.state[p]  # always empty
                # grad_shape = grad.shape
                factored, use_first_moment = self._get_options(group, p_shape)
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        if sharding_spec.sharding_sequence[0] == 'R': # Col Parallel
                            state["exp_avg_sq_row"] = torch.zeros(param_height).to(grad)  # [H:4096]
                            state["exp_avg_sq_col"] = torch.zeros(param_width_parallel).to(grad)  # [W/N:2048]
                        
                        if sharding_spec.sharding_sequence[-1] == 'R': # Row Parallel
                            state["exp_avg_sq_row"] = torch.zeros(param_height_parallel).to(grad)  # [H/N:2048]
                            state["exp_avg_sq_col"] = torch.zeros(param_width).to(grad)  # [W:4096]
                            
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.to(self.localRank)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()
                state["step"] += 1
                lr = self._get_lr(group, state)
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:  
                    # ==============================
                    # First Dim is R, Last Dim is S{} means split dim -1  ---> 
                    # Coloum Parallel ---> sq_row need Do (col) Reduce
                    # ==============================
                    if sharding_spec.sharding_sequence[0] == 'R': 
                        update_reshape = update.view(-1, param_width_parallel)
                        # print(f"grad {grad}")
                        grad_reshape = grad.view(-1, param_width_parallel)
                        exp_avg_sq_row = state["exp_avg_sq_row"] # [H]
                        exp_avg_sq_col = state["exp_avg_sq_col"] # [W/N]
                        exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                        exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                        dist.all_reduce(exp_avg_sq_row, group=self.tensor_parallel_group)
                        exp_avg_sq_row.div_(self.tensor_parallel_size)
                        update_reshape = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                        update_reshape.mul_(grad_reshape)
                        update = update_reshape.flatten()
                    # ==============================
                    # Last Dim is R, First Dim is S{} means split dim 0  --->
                    # Row Parallel ---> sq_col need Do (row) Reduce
                    # ==============================
                    elif sharding_spec.sharding_sequence[-1] == 'R':
                        update_reshape = update.view(-1, param_width)
                        grad_reshape = grad.view(-1, param_width)
                        exp_avg_sq_row = state["exp_avg_sq_row"] # [H/N]
                        exp_avg_sq_col = state["exp_avg_sq_col"] # [W]
                        exp_avg_sq_row.mul_(beta2t).add_(update_reshape.mean(dim=-1), alpha=(1.0 - beta2t))
                        exp_avg_sq_col.mul_(beta2t).add_(update_reshape.mean(dim=-2), alpha=(1.0 - beta2t))
                        # reduce col
                        dist.all_reduce(exp_avg_sq_col, group=self.tensor_parallel_group)
                        exp_avg_sq_col.div_(self.tensor_parallel_size)
                        # gather row
                        exp_avg_sq_row_gather = _gather(exp_avg_sq_row, self.tensor_parallel_group)
                        sq_row_meam = exp_avg_sq_row_gather.mean(dim=-1, keepdim=True)
                        update_reshape = self._approx_sq_grad_row_parallel(exp_avg_sq_row, exp_avg_sq_col, sq_row_meam)
                        update_reshape.mul_(grad_reshape)
                        update = update_reshape.flatten()
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                # (Line No.8) RMS
                # update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)
            
                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))
                
                p_data_fp32.add_(-update).flatten()
                
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
                
        return loss


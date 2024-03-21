import os
import sys
sys.path.append('./colossalai/nn/optimizer/')
import time
from prettytable import PrettyTable

import torch
from torch import nn
import torch.distributed as dist

from adafactor import Adafactor
from distributed_adafactor import DistributedAdaFactor

from colossalai.tensor.d_tensor import (
    distribute_tensor,
    ShardingSpec,
)
import colossalai
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import parameterize
from colossalai.shardformer.layer._operation import _gather
# from colossalai.nn.optimizer import DistributedAdaFactor

# # Set weight bias
# def setup_param_groups(bert_model: nn.Module) -> list:
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": 0.1,
#         },
#         {
#             "params": [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#     return optimizer_grouped_parameters

# # Set Gradent
# def set_grad(model: nn.Module, torch_model: nn.Module, g_dtype: torch.dtype) -> None:
#     for p, torch_p in zip(model.parameters(), torch_model.parameters()):
#         torch_p.grad = torch.rand_like(torch_p)
#         # avoid inconsistent grad and param dtype error
#         orig_p = p.data
#         p.data = torch_p.grad.clone().to(g_dtype)
#         p.grad = p.data
#         p.data = orig_p

# dist env
def init_dist():
    rank = int(os.environ['RANK']) 
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)

def correctness_verify(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float32:
        rtol = 1e-05
        atol = 1e-05
    elif dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3
    
    return torch.all(tensor1.isclose(tensor2, rtol=rtol, atol=atol, equal_nan=True))
    # return torch.testing.assert_close(tensor1, tensor2,  rtol=1e-05, atol=1e-05, equal_nan=True)
    
def error_idx(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float32:
        rtol = 1e-05
        atol = 1e-05
    elif dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3
    return torch.isclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=True)

def get_time():
    torch.cuda.synchronize()
    return time.time()

@parameterize("dtype", [torch.float32,torch.float16, torch.bfloat16])
def main(dtype):
    # ==============================
    # torch distributed init
    # ==============================
    torch.manual_seed(0)
    device = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    tensor_parallel_size = world_size
    # init_dist()
    torch.set_default_dtype(dtype)
    if device == 0:
        print(f"Dtype {dtype}\n")
    
    # ==============================
    # Col Parallel Param Info
    # ==============================
    H, W = 4096, 4096
    model_col = nn.Linear(H, W).to(device) # Col parallel weight  
    
    # ==============================
    # Param init
    # ==============================
    # Col Parallel 
    # Weight split along last dim
    # base param
    # layer shape [H, W], then
    # weight [W, H] [4, 2]
    # bias [W]  [4]
    weight, bias = model_col.weight, model_col.bias
    # physical_mesh_id: torch.Tensor[0,1,2,3]
    # logical_mesh_id: (DP size, TP size); WORLD SIZE = DP size, TP size; 2 GPU view as (1,2); 4 GPU view as (1,4) or (2,2)
    device_mesh = DeviceMesh(torch.Tensor([i for i in range(world_size)]), (1, tensor_parallel_size), init_process_group=True)
    sharding_spec = ShardingSpec(dim_size=weight.dim(), dim_partition_dict={weight.dim() - 1: [1]})
    weight_shard = distribute_tensor(weight, device_mesh, sharding_spec)
    # local_weight [W*H/N] [4*1]
    # local_bias [W]  [4]
    local_weight_flatten = nn.Parameter(weight_shard.clone().flatten().requires_grad_(True))
    local_bias_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
    params_shape = {id(local_weight_flatten): weight.shape, id(local_bias_flatten): bias.shape}
    sharding_spec_dict = {id(local_weight_flatten): sharding_spec, id(local_bias_flatten): None}

    
    # ==============================
    # Adafactor Base (For Coloumn Parallel)
    # ==============================
    torch.cuda.synchronize()
    optimizer_base = Adafactor([weight, bias])
    optimizer_base.zero_grad()
    weight.grad = torch.rand_like(weight)
    bias.grad = torch.rand_like(bias)
    optimizer_base.step()

    # ==============================
    # DistributedAdafactor (Coloumn Parallel)
    # ==============================
    torch.cuda.synchronize()
    optimizer_zero2 = DistributedAdaFactor([local_weight_flatten, local_bias_flatten])
    optimizer_zero2.setup_distribute(device_mesh=device_mesh, sharding_spec_dict=sharding_spec_dict, param_shape = params_shape)
    optimizer_zero2.zero_grad()
    local_weight_flatten.grad = distribute_tensor(weight.grad, device_mesh, sharding_spec).clone().flatten()
    local_bias_flatten.grad = bias.grad.clone().flatten()
    optimizer_zero2.step()
    
    # ==============================
    # Correctness Verify (Column Parallel)
    # ==============================
    # tensor parallel & flatten view &gather data (Coloumn Parallel)
    torch.cuda.synchronize()
    reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
    gather_flatten_weight = _gather(input_ = reshape_flatten_weight.data, dim=-1, process_group=device_mesh.get_process_group(axis=1)) # gather
    weight_correctness = correctness_verify(weight.data, gather_flatten_weight, dtype)
    bias_correctness = correctness_verify(bias.data, local_bias_flatten.data, dtype)
    if weight_correctness:
        print(f"Distributed weight (Column Parallel) correctness Pass")
    else:
        print(f"Distributed weight (Column Parallel) inclued incorrectness value")
        weight_err_idx = error_idx(weight.data, gather_flatten_weight.data, dtype)
        print(f"Distributed weight (Column Parallel) err idx {weight_err_idx}")
    if bias_correctness:
        print(f"Distributed bias (Column Parallel) correctness Pass")
    else:
        print(f"Distributed bias (Column Parallel) inclued incorrectness value")
        bias_err_idx = error_idx(bias.data, local_bias_flatten.data, dtype)
        print(f"Distributed bias (Column Parallel) err idx {bias_err_idx}")
        
    # ==============================
    # Runtime Test (col Parallel)
    # ==============================
    niter = 10
    base_start, base_end, base_runtime = 0, 0, 0
    zero_start, zero_end, zero_runtime, zero_best_runtime = 0, 0, 0, float('inf')
    table = PrettyTable(['Version', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    for i in range(0, niter):
        # Base Adafactor
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        base_start = get_time()
        optimizer_base.step()
        base_end = get_time()
        
        # Distributed Adafactor
        optimizer_zero2.zero_grad()
        local_weight_flatten.grad = distribute_tensor(weight.grad, device_mesh, sharding_spec).clone().flatten()
        local_bias_flatten.grad = bias.grad.clone().flatten()
        zero_start = get_time()
        optimizer_zero2.step()
        zero_end = get_time()
    
        reshape_flatten_weight = local_weight_flatten.view(-1, H // tensor_parallel_size) # reshape
        gather_flatten_weight = _gather(input_ = reshape_flatten_weight.data, dim=-1, process_group=device_mesh.get_process_group(axis=1)) # gather
        
        torch.cuda.synchronize()
        v3_weight_correctness = correctness_verify(weight.data, gather_flatten_weight, dtype)
        v3_bias_correctness = correctness_verify(bias.data, local_bias_flatten.data, dtype)
        
        
        print(f"iter {i}")
        if v3_weight_correctness:
            print(f"Distributed weight (Column Parallel) correctness Pass")
        else:
            print(f"Distributed weight (Column Parallel) inclued incorrectness value")
            weight_err_idx = error_idx(weight.data, gather_flatten_weight.data, dtype)
            print(f"Distributed weight (Column Parallel) err idx {weight_err_idx}")
                
        if v3_bias_correctness:
            print(f"Distributed bias (Column Parallel) correctness Pass")
        else:
            print(f"Distributed bias (Column Parallel) inclued incorrectness value")
            bias_err_idx = error_idx(bias.data, local_bias_flatten.data, dtype)
            print(f"Distributed bias (Column Parallel) err idx {bias_err_idx}")


        base_runtime += base_end - base_start
        zero_runtime  += zero_end - zero_start
        zero_best_runtime= min(zero_best_runtime, zero_runtime)

    table = PrettyTable(['Version','iter', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Avg Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    table.add_row(["AdaFactor", niter, weight.shape, bias.shape, (base_runtime / niter) * 10.0**3, None, None])
    table.add_row(["DistributedAdaFactor (Col Parallel)", niter, weight.shape, bias.shape, (zero_runtime / niter) * 10.0**3, base_runtime/zero_runtime ,base_runtime/zero_best_runtime])
    
    print(table)
    
    
        
    # ==============================
    # Row Parallel Param Info
    # ==============================
    # Weight split along First dim
    model_row = nn.Linear(W, H).to(device) # Row parallel weight  
    weight_row, bias_row = model_row.weight, model_row.bias
    device_mesh_row = DeviceMesh(torch.Tensor([i for i in range(world_size)]), (1, tensor_parallel_size), init_process_group=True)
    sharding_spec_row = ShardingSpec(dim_size=weight.dim(), dim_partition_dict={0: [1]})
    weight_row_shard = distribute_tensor(weight_row, device_mesh_row, sharding_spec_row)
    local_weight_row_flatten = nn.Parameter(weight_row_shard.clone().flatten().requires_grad_(True))
    local_bias_row_flatten = nn.Parameter(bias_row.clone().flatten().requires_grad_(True))
    params_shape_row = {id(local_weight_row_flatten): weight.shape, id(local_bias_row_flatten): bias.shape}
    sharding_spec_row_dict = {id(local_weight_row_flatten): sharding_spec_row, id(local_bias_row_flatten): None}

    # ==============================
    # Adafactor Base (For Row Parallel)
    # ==============================
    torch.cuda.synchronize()
    optimizer_base = Adafactor([weight_row, bias_row])
    optimizer_base.zero_grad()
    weight_row.grad = torch.rand_like(weight_row)
    bias_row.grad = torch.rand_like(bias_row)
    optimizer_base.step()

    # ==============================
    # DistributedAdafactor (Row Parallel)
    # ==============================
    torch.cuda.synchronize()
    optimizer_zero2 = DistributedAdaFactor([local_weight_row_flatten, local_bias_row_flatten])
    optimizer_zero2.setup_distribute(device_mesh=device_mesh_row, sharding_spec_dict=sharding_spec_row_dict, param_shape = params_shape_row)
    optimizer_zero2.zero_grad()
    local_weight_row_flatten.grad = distribute_tensor(weight_row.grad, device_mesh_row, sharding_spec_row).clone().flatten()
    local_bias_row_flatten.grad = bias_row.grad.clone().flatten()
    optimizer_zero2.step()

    # ==============================
    # Correctness Verify (Row Parallel)
    # ==============================
    torch.cuda.synchronize()
    gather_flatten_weight_row = _gather(input_=local_weight_row_flatten.data,dim=-1, process_group=device_mesh_row.get_process_group(axis=1)) # gather
    reshape_flatten_weight_row = gather_flatten_weight_row.view(-1, W) # reshape


    weight_correctness = correctness_verify(weight_row.data, reshape_flatten_weight_row, dtype)
    bias_correctness = correctness_verify(bias_row.data, local_bias_row_flatten.data, dtype)
    if weight_correctness:
        print(f"Distributed weight (Row Parallel) correctness Pass")
    else:
        print(f"Distributed weight (Row Parallel) inclued incorrectness value")
        weight_err_idx = error_idx(weight_row.data, reshape_flatten_weight_row.data, dtype)
        print(f"Distributed weight err idx {weight_err_idx}")
    if bias_correctness:
        print(f"Distributed bias (Row Parallel) correctness Pass")
    else:
        print(f"Distributed bias (Row Parallel) inclued incorrectness value")
        bias_err_idx = error_idx(bias_row.data, local_bias_row_flatten.data, dtype)
        print(f"Distributed bias (Row Parallel) err idx {bias_err_idx}")
            
    # ==============================
    # Runtime Test (Row Parallel)
    # ==============================
    niter = 10
    base_start, base_end, base_runtime = 0, 0, 0
    zero_start, zero_end, zero_runtime, zero_best_runtime = 0, 0, 0, float('inf')
    table = PrettyTable(['Version', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    for i in range(0, niter):
        # Base Adafactor
        optimizer_base.zero_grad()
        weight_row.grad = torch.rand_like(weight_row)
        bias_row.grad = torch.rand_like(bias_row)
        base_start = get_time()
        optimizer_base.step()
        base_end = get_time()
        
        # Distributed Adafactor
        optimizer_zero2.zero_grad()
        local_weight_row_flatten.grad = distribute_tensor(weight_row.grad, device_mesh_row, sharding_spec_row).clone().flatten()
        local_bias_row_flatten.grad = bias_row.grad.clone().flatten()
        zero_start = get_time()
        optimizer_zero2.step()
        zero_end = get_time()
    
        torch.cuda.synchronize()
    
        gather_flatten_weight_row = _gather(input_=local_weight_row_flatten.data,dim=-1, process_group=device_mesh_row.get_process_group(axis=1)) # gather
        reshape_flatten_weight_row = gather_flatten_weight_row.view(-1, W) # reshape

        weight_correctness = correctness_verify(weight_row.data, reshape_flatten_weight_row, dtype)
        bias_correctness = correctness_verify(bias_row.data, local_bias_row_flatten.data, dtype)
        
        
        print(f"iter {i}")
        if weight_correctness:
            print(f"Distributed weight (Row Parallel) correctness Pass")
        else:
            print(f"Distributed weight (Row Parallel) inclued incorrectness value")
            weight_err_idx = error_idx(weight_row.data, reshape_flatten_weight_row.data, dtype)
            print(f"Distributed weight (Row Parallel) err idx {weight_err_idx}")
                
        if bias_correctness:
            print(f"Distributed bias (Row Parallel) correctness Pass")
        else:
            print(f"Distributed bias (Row Parallel) inclued incorrectness value")
            bias_err_idx = error_idx(bias_row.data, local_bias_row_flatten.data, dtype)
            print(f"Distributed bias (Row Parallel) err idx {bias_err_idx}")


        base_runtime += base_end - base_start
        zero_runtime  += zero_end - zero_start
        zero_best_runtime= min(zero_best_runtime, zero_runtime)

    table = PrettyTable(['Version','iter', 'weight shape', 'bias shape', 'Avg runtime(ms)',
                        'Avg Speed Up Rate', 'Best Speed Up Rate'], float_format='.2')
    table.add_row(["AdaFactor", niter, weight_row.shape, bias_row.shape, (base_runtime / niter) * 10.0**3, None, None])
    table.add_row(["DistributedAdaFactor (Row Parallel)", niter, weight_row.shape, bias_row.shape, (zero_runtime / niter) * 10.0**3, base_runtime/zero_runtime ,base_runtime/zero_best_runtime])
    
    print(table)


if __name__ == "__main__":
    init_dist()
    main()
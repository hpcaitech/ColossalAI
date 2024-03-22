import pytest
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.testing import assert_close

import colossalai
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor import (
    distribute_tensor,
    sharded_tensor_to_param,
    shard_rowwise,
    shard_colwise,
    get_layout,
    get_sharding_spec
)
from colossalai.shardformer.layer._operation import _gather


import sys
sys.path.append('./colossalai/nn/optimizer/')
from adafactor import Adafactor
from distributed_adafactor import DistributedAdaFactor

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
    
    # return torch.all(tensor1.isclose(tensor2, rtol=rtol, atol=atol, equal_nan=True))
    assert_close(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=True)

@parameterize("dtype", [torch.float32, torch.float16, torch.bfloat16]) # , torch.float16, torch.bfloat16
def exam_dist_adafactor_step(dtype: torch.dtype):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    tensor_parallel_size = world_size
    torch.set_default_dtype(dtype)
    set_seed(42)
    
    # ==============================
    # Base Case
    # ==============================
    H, W = 4, 4
    model_col = nn.Linear(H, W).to(local_rank) # Col parallel weight  
    weight, bias = model_col.weight, model_col.bias
    device_mesh = DeviceMesh(torch.Tensor([i for i in range(world_size)]), (1, tensor_parallel_size), init_process_group=True)
    
    # ==============================
    # Col Parallel
    # ==============================
    weight_col_shard = shard_colwise(weight.clone(), device_mesh.get_process_group(axis=1))
    weight_col_shard_layout = get_layout(weight_col_shard) # Layout info weight_col_shard_layout.global_shape
    weight_col_shard_shard_spec = get_sharding_spec(weight_col_shard) # Shard spec
    weight_col_shard_flatten = nn.Parameter(weight_col_shard.clone().flatten().requires_grad_(True))
    bias_col_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
    col_params_shape = {id(weight_col_shard_flatten): weight_col_shard_layout.global_shape, id(bias_col_flatten): bias.shape} 
    col_sharding_spec_dict = {id(weight_col_shard_flatten): weight_col_shard_shard_spec, id(bias_col_flatten): None}
    
    # ==============================
    # Row Parallel 
    # ==============================
    weight_row_shard = shard_rowwise(weight.clone(), device_mesh.get_process_group(axis=1))
    weight_row_shard_layout = get_layout(weight_row_shard) # Layout info weight_row_shard_layout.global_shape
    weight_row_shard_shard_spec = get_sharding_spec(weight_row_shard) # Shard spec
    weight_row_shard_flatten = nn.Parameter(weight_row_shard.clone().flatten().requires_grad_(True)) # flatten input(not dtensor) to optimizer
    bias_row_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
    row_params_shape = {id(weight_row_shard_flatten): weight_row_shard_layout.global_shape, id(bias_row_flatten): bias.shape} 
    row_sharding_spec_dict = {id(weight_row_shard_flatten): weight_row_shard_shard_spec, id(bias_row_flatten): None}

    # ==============================
    # Init Optimizer 
    # ==============================
    
    # base 
    optimizer_base = Adafactor([weight, bias])
    
    # col parallel
    optimizer_cp = DistributedAdaFactor([weight_col_shard_flatten, bias_col_flatten])
    optimizer_cp.setup_distribute(device_mesh=device_mesh, sharding_spec_dict=col_sharding_spec_dict, param_shape = col_params_shape)
    
    # row parallel
    optimizer_rp = DistributedAdaFactor([weight_row_shard_flatten, bias_row_flatten])
    optimizer_rp.setup_distribute(device_mesh=device_mesh, sharding_spec_dict=row_sharding_spec_dict, param_shape = row_params_shape)
    
    N_STEPS = 10
    for _ in range(N_STEPS):
        # base step
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        optimizer_base.step()
        
        # col parallel step
        optimizer_cp.zero_grad()
        weight_col_shard_flatten.grad = distribute_tensor(weight.grad, device_mesh, weight_col_shard_shard_spec).clone().flatten()
        bias_col_flatten.grad = bias.grad.clone().flatten()
        optimizer_cp.step()
        
        # row parallel step
        optimizer_rp.zero_grad()
        weight_row_shard_flatten.grad = distribute_tensor(weight.grad, device_mesh, weight_row_shard_shard_spec).clone().flatten()
        bias_row_flatten.grad = bias.grad.clone().flatten()
        optimizer_rp.step()
        
        # gather result
        weight_col_gather = _gather(input_=weight_col_shard_flatten.data.view(-1, H // tensor_parallel_size),dim=-1, process_group=device_mesh.get_process_group(axis=1)) # gather
        weight_row_gather = _gather(input_=weight_row_shard_flatten.data,dim=-1, process_group=device_mesh.get_process_group(axis=1)).view(-1, W) # gather
        
        
        # verify
        col_correct = correctness_verify(weight.data, weight_col_gather.data, dtype)
        row_correct = correctness_verify(weight.data, weight_row_gather.data, dtype)
        
        # print(f"col corrness {col_correct}  row correct {row_correct}")
    
def run_dist(rank, world_size, port):
    config = {}
    # colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    init_dist()
    exam_dist_adafactor_step()

@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_adafactor():
    spawn(run_dist, nprocs=1)
    
if __name__ == "__main__":
    test_dist_adafactor()
    pass
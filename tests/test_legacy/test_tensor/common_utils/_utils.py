import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.testing import assert_close

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.tensor import ComputePattern, ComputeSpec, ShardSpec


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1) == True


def replace_parameter_add_grad(layer, weight=None, bias=None):
    if weight is not None:
        delattr(layer, "weight")
        setattr(layer, "weight", weight)
        layer.weight.requires_grad = True
    if bias is not None:
        delattr(layer, "bias")
        setattr(layer, "bias", bias)
        layer.bias.requires_grad = True


def broadcast_tensor_chunk(tensor, chunk_size=1, local_rank=0):
    dist.broadcast(tensor, src=0)
    tensor_chunk = torch.chunk(tensor, chunk_size, dim=-1)[local_rank]
    return tensor_chunk.clone()


def tensor_equal(t_a: torch.Tensor, t_b: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-1):
    assert_close(t_a, t_b, rtol=rtol, atol=atol)
    return True


def tensor_shard_equal(
    tensor: torch.Tensor, shard: torch.Tensor, rank: int, world_size: int, rtol: float = 1e-3, atol: float = 1e-1
):
    assert tensor.ndim == shard.ndim
    if tensor.shape == shard.shape:
        return tensor_equal(tensor, shard, rtol, atol)
    else:
        dims_not_eq = torch.nonzero(torch.tensor(tensor.shape) != torch.tensor(shard.shape))
        if dims_not_eq.numel() == 1:
            # 1D shard
            dim = dims_not_eq.item()
            if world_size is None:
                world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
            if rank is None:
                rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
            return tensor_equal(tensor.chunk(world_size, dim)[rank], shard, rtol, atol)
        else:
            raise NotImplementedError


def split_param_single_dim_tp1d(dim, param, pg):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    if param.process_group.tp_world_size() == 1:
        param.set_process_group(pg)
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param, pg):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param, pg):
    split_param_single_dim_tp1d(-1, param, pg)


def debug_print(ranks, *args):
    if dist.get_rank() in ranks:
        print(*args)
    dist.barrier()

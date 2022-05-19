import os
import random
import numpy as np
import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1) == True


def replace_parameter_add_grad(layer, weight=None, bias=None):
    if weight is not None:
        delattr(layer, 'weight')
        setattr(layer, 'weight', weight)
        layer.weight.requires_grad = True
    if bias is not None:
        delattr(layer, 'bias')
        setattr(layer, 'bias', bias)
        layer.bias.requires_grad = True


def broadcast_tensor_chunk(tensor, chunk_size=1, local_rank=0):
    dist.broadcast(tensor, src=0)
    tensor_chunk = torch.chunk(tensor, chunk_size, dim=-1)[local_rank]
    return tensor_chunk.clone()


def tensor_equal(A, B):
    return torch.allclose(A, B, rtol=1e-3, atol=1e-1)


def tensor_shard_equal(tensor: torch.Tensor, shard: torch.Tensor):
    assert tensor.ndim == shard.ndim
    if tensor.shape == shard.shape:
        return tensor_equal(tensor, shard)
    else:
        dims_not_eq = torch.nonzero(torch.tensor(tensor.shape) != torch.tensor(shard.shape))
        if dims_not_eq.numel() == 1:
            # 1D shard
            dim = dims_not_eq.item()
            world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
            rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
            return tensor_equal(tensor.chunk(world_size, dim)[rank], shard)
        else:
            raise NotImplementedError

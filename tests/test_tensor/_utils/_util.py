import torch
import torch.distributed as dist


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

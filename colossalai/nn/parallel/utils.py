import torch
import torch.distributed as dist

from colossalai.gemini.chunk import Chunk
from colossalai.tensor import ColoTensor
from colossalai.utils import get_current_device


def get_temp_total_chunk_on_cuda(chunk: Chunk):
    if chunk.is_gathered:
        return chunk.cuda_global_chunk

    if chunk.cuda_shard is not None:
        shard_temp = chunk.cuda_shard
    else:
        shard_temp = chunk.cpu_shard.to(get_current_device())

    total_temp = torch.zeros(chunk.chunk_size, dtype=chunk.dtype, device=get_current_device())
    gather_list = list(torch.chunk(input=total_temp, chunks=chunk.pg_size, dim=0))
    dist.all_gather(tensor_list=gather_list, tensor=shard_temp, group=chunk.torch_pg)

    return total_temp


def _add_param(model, name, param):
    name_list = name.split('.')
    module = model._modules[name_list[0]]
    for i in range(1, len(name_list) - 1):
        module = module._modules[name_list[i]]
    module._parameters[name_list[-1]] = param


def convert_to_torch_module(gemini_ddp_model) -> torch.nn.Module:
    """convert_to_torch_module

    Args:
        gemini_ddp_model (GeminiDDP): a gemini ddp model

    Returns:
        torch.nn.Module: a torch model contains the params of gemini_ddp_model
    """
    module = gemini_ddp_model.module

    for n, p in module.named_parameters():
        if isinstance(p, ColoTensor):
            p.to_replicate_()
            _add_param(module, n, p.data)

    return module

import torch
import torch.distributed as dist

from colossalai.gemini.chunk import Chunk
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


# TODO() not work for module where two params share the same tensor.
def _add_param(model, name, param):
    name_list = name.split('.')
    module = model._modules[name_list[0]]
    for i in range(1, len(name_list) - 1):
        module = module._modules[name_list[i]]
    module._parameters[name_list[-1]] = param


def convert_to_torch_module(gemini_ddp_model: 'GeminiDDP') -> torch.nn.Module:
    """convert_to_torch_module

    Args:
        gemini_ddp_model (GeminiDDP): a gemini ddp model

    Returns:
        torch.nn.Module: a torch model contains the params of gemini_ddp_model
    """
    from colossalai.nn.parallel import GeminiDDP
    assert isinstance(gemini_ddp_model, GeminiDDP)
    module = gemini_ddp_model.module

    # replace ColoTensor to torch.nn.Tensor in module
    for n, p in gemini_ddp_model.torch_named_parameters():
        _add_param(module, n, p)

    return module

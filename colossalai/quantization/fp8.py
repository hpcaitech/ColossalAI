import torch
import torch.distributed as dist


def cast_to_fp8(inp: torch.Tensor, scale=None, fp8_format="e4m3") -> (torch.Tensor, torch.Tensor):
    r"""
    casting torch Tensor into specified fp8 tensor with per-channel scaling or per-tensor scaling.
    Args:
        inp: input torch Tensor, should be in torch.FloatTensor, torch.HalfTensor, torch.BFloat16Tensor.
        scale: scaling factor for fp8 casting. If it is None, then it is computed automatically. Per-channel scaling
        is applied if input tensor is 2 dimension, otherwise, per-tensor scaling is applied.
        fp8_format: e4m3 or e5m2

    Returns:
        Tuples: A tuple (fp8_tensor, scale)
    """
    if inp.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        return inp
    fp8_type = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    if inp.dim() == 2:
        if scale is None:
            per_channel_max = inp.abs().max(dim=-1).values
            scale = per_channel_max
        scale_inv = 1.0 / scale
        scale_inv = scale_inv[:, None]
        ret = (scale_inv * inp).to(fp8_type)
    else:
        if scale is None:
            per_tensor_max = inp.abs().max()
            scale = per_tensor_max
        scale_inv = 1.0 / scale
        ret = (scale_inv * inp).to(fp8_type)

    return ret, scale


def cast_from_fp8(inp: torch.Tensor, scale: torch.Tensor, ret_type: torch.dtype) -> torch.Tensor:
    r"""

    Args:
        inp: should be a fp8 torch tensor in one of the types: [torch.float8_e4m3fn, torch.float8_e5m2].
        scale: scaling factor returned by cast_to_fp8 function.
        ret_type: the datatype of the returned tensor.

    Returns:
        torch.Tensor
    """
    if inp.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return inp
    if inp.dim() == 2:
        ret = scale[:, None] * inp.to(ret_type)
    else:
        ret = scale * inp.to(ret_type)
    return ret


def all_reduce_fp8(tensor: torch.Tensor, fp8_format="e4m3") -> None:
    r"""
    This is an in-place operation for compressed all_reduce using fp8.
    It works like dist.all_reduce but during communication the data is cast to fp8 format.

    Args:
        tensor: torch.Tensor in fp32, fp16, bf16 datatype.
        fp8_format: e4m3 or e5m2

    Returns:
        None
    """

    world_size = dist.get_world_size()
    dist.get_rank()
    input_type = tensor.dtype
    input_shape = tensor.shape
    input_device = tensor.device
    input_size = tensor.numel()
    tensor = tensor.flatten()

    fp8_type = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2

    ret, scale = cast_to_fp8(tensor, fp8_format=fp8_format)

    inp = ret.view(torch.uint8)
    input_chunks = list(torch.chunk(inp, world_size, dim=0))
    if dist.get_rank() == world_size - 1:
        output_chunks = [torch.empty_like(input_chunks[-1]) for _ in range(world_size)]
    else:
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]
    dist.all_to_all(output_chunks, input_chunks)
    scale_list = [torch.ones(1, dtype=scale.dtype, device=input_device) for _ in range(world_size)]
    dist.all_gather(scale_list, scale)
    summed_out = torch.zeros_like(output_chunks[0]).to(input_type)
    for scale, out in zip(scale_list, output_chunks):
        out = out.view(fp8_type)
        summed_out += cast_from_fp8(out, scale, input_type)

    summed_out_fp8, scale = cast_to_fp8(summed_out, fp8_format=fp8_format)
    dist.all_gather(scale_list, scale)

    tensor_list = list(torch.chunk(torch.empty(input_size, device=input_device, dtype=torch.uint8), world_size, dim=0))
    dist.all_gather(tensor_list, summed_out_fp8.view(torch.uint8))
    for i in range(world_size):
        tensor_list[i] = tensor_list[i].view(fp8_type).to(input_type) * scale_list[i]
    tensor_out = torch.cat(tensor_list, dim=0)
    tensor.data = tensor_out.view(input_shape).to(input_type)

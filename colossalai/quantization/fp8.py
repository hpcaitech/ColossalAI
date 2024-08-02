from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def cast_to_fp8(inp: torch.Tensor, fp8_format="e4m3") -> (torch.Tensor, torch.Tensor):
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
        raise TypeError("Only float16, bfloat16, and float32 are allowed.")

    fp8_type = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2
    fp8_max = torch.finfo(fp8_type).max

    if inp.dim() == 2:
        per_channel_max = inp.abs().max(dim=-1).values.float()
        per_channel_max = torch.where(per_channel_max > 0, per_channel_max, 1.0)
        scale = fp8_max / per_channel_max[:, None]
    else:
        per_tensor_max = inp.abs().max().float()
        per_tensor_max = torch.where(per_tensor_max > 0, per_tensor_max, 1.0)
        scale = fp8_max / per_tensor_max

    scale_inv = 1.0 / scale
    ret = (scale * inp.float()).to(fp8_type)
    return ret, scale_inv


def cast_from_fp8(inp: torch.Tensor, scale_inv: torch.Tensor, ret_type: torch.dtype) -> torch.Tensor:
    r"""
    Args:
        inp: should be a fp8 torch tensor in one of the types: [torch.float8_e4m3fn, torch.float8_e5m2].
        scale: scaling factor returned by cast_to_fp8 function.
        ret_type: the datatype of the returned tensor.
    Returns:
        torch.Tensor
    """
    if inp.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise TypeError("Only float8_e4m3fn and float8_e5m2 are allowed.")

    if inp.dim() == 2:
        ret = scale_inv[:, None] * inp.float()
    else:
        ret = scale_inv * inp.float()
    return ret.to(ret_type)


def all_reduce_fp8(tensor: torch.Tensor, fp8_format="e5m2", group=None) -> None:
    r"""
    This is an in-place operation for compressed all_reduce using fp8.
    It works like dist.all_reduce but during communication the data is cast to fp8 format.
    Args:
        tensor: torch.Tensor in fp32, fp16, bf16 datatype.
        fp8_format: e4m3 or e5m2
    Returns:
        None
    """

    world_size = dist.get_world_size(group=group)
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
    dist.all_to_all(output_chunks, input_chunks, group=group)
    scale_list = [torch.ones(1, dtype=scale.dtype, device=input_device) for _ in range(world_size)]
    dist.all_gather(scale_list, scale, group=group)
    summed_out = torch.zeros_like(output_chunks[0]).to(input_type)
    for scale, out in zip(scale_list, output_chunks):
        out = out.view(fp8_type)
        summed_out += cast_from_fp8(out, scale, input_type)

    summed_out_fp8, scale = cast_to_fp8(summed_out, fp8_format=fp8_format)
    dist.all_gather(scale_list, scale, group=group)

    tensor_list = list(torch.chunk(torch.empty(input_size, device=input_device, dtype=torch.uint8), world_size, dim=0))
    dist.all_gather(tensor_list, summed_out_fp8.view(torch.uint8), group=group)
    for i in range(world_size):
        tensor_list[i] = tensor_list[i].view(fp8_type).to(input_type) * scale_list[i]
    tensor_out = torch.cat(tensor_list, dim=0)
    tensor.data = tensor_out.view(input_shape).to(input_type)


def cast_to_fp8_pipeline(inp: Any) -> None:
    """
    Cast the hidden_states tensor of inp object to fp8 format before p2p communication in pipeline.
    The activations tensor is indexed by 'hidden_states' in the inp dict.
    After FP8 casting, the resulting tensor is saved as float16 or bfloat16 format but the size becomes halved.
    Metadata such as fp8_scale is saved into inp dict for communication.
    """
    if inp is None:
        return
    # In pipeline parallelism, when inp is torch.Tensor, it only contains one element, thus can be omitted.
    if type(inp) == torch.Tensor:
        return

    assert "hidden_states" in inp, "required by pipeline parallelism."
    inp_tensor = inp["hidden_states"]

    min_val, max_val = inp_tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())

    finfo = torch.finfo(torch.float8_e4m3fn)
    if amax > finfo.max:
        fp8_type = torch.float8_e5m2
        fp8_view_type = torch.float16
    else:
        fp8_type = torch.float8_e4m3fn
        fp8_view_type = torch.bfloat16

    finfo = torch.finfo(fp8_type)
    scale = torch.tensor(1.0).to(inp_tensor.device) if amax == 0.0 else finfo.max / amax.float()
    q_tensor = inp_tensor.data.float() * scale
    # Todo: Currently we use fp8_view_type <float16, bfloat16> to indicate which fp8 format is used. This is a temporary workaround due to 'Only support tensor for fast send'.
    #  inp_tensor needs to be a float datatype to avoid error during gradient placement.
    inp_tensor.data = q_tensor.to(fp8_type).view(fp8_view_type)

    inp["fp8_scale"] = scale.float().reciprocal()


def cast_from_fp8_pipeline(inp: Any, del_metadata=True) -> None:
    """
    Cast the FP8 encoded hidden_states tensor back to original dtype after p2p communication in pipeline.
    del_metadata = False is useful when this function is called before p2p communication.
    """
    if inp is None:
        return
    if type(inp) == torch.Tensor:
        return

    assert "hidden_states" in inp, "required by pipeline parallelism."
    inp_tensor = inp["hidden_states"]
    scale = inp["fp8_scale"]

    fp8_view_type = inp_tensor.dtype
    if fp8_view_type == torch.float16:
        fp8_type = torch.float8_e5m2
    elif fp8_view_type == torch.bfloat16:
        fp8_type = torch.float8_e4m3fn
    else:
        raise TypeError("Only float16, bfloat16 are implemented.")

    inp_tensor.data = inp_tensor.data.view(fp8_type).to(torch.float16) * scale

    if del_metadata:
        del inp["fp8_scale"]


def reduce_scatter_fp8(output: torch.Tensor, input_list, group, fp8_format="e5m2") -> None:
    r"""
    This is an in-place operation for compressed reduce_scatter using fp8.
    It works like dist.reduce_scatter but during communication the data is cast to fp8 format.

    Args:
        tensor: torch.Tensor in fp32, fp16, bf16 datatype.
        fp8_format: e4m3 or e5m2

    Returns:
        None
    """

    input_type = output.dtype

    fp8_type = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2
    scale_list = []
    cast_input_list = []
    output_chunks = []
    output_scale_list = []
    for input in input_list:
        ret, scale = cast_to_fp8(input, fp8_format=fp8_format)
        scale_list.append(scale)
        ret = ret.view(torch.uint8)
        cast_input_list.append(ret)
        output_chunks.append(torch.empty_like(ret))
        output_scale_list.append(torch.empty_like(scale))
    dist.all_to_all(output_chunks, cast_input_list, group=group)
    dist.all_to_all(output_scale_list, scale_list, group=group)

    summed_out = torch.zeros_like(output_chunks[0]).to(input_type)
    for scale, out in zip(output_scale_list, output_chunks):
        out = out.view(fp8_type)
        summed_out += cast_from_fp8(out, scale, input_type)
    output.data = summed_out


def split_chunk_by_channel(
    chunk: torch.Tensor, channel_size: int, num_channels: int, rank: int = 0, world_size: int = 1
):
    offset = chunk.numel() * rank
    end = offset + chunk.numel()
    break_points = [x for x in range(0, channel_size * num_channels + 1, channel_size) if offset <= x <= end]
    if len(break_points) == 0 or break_points[0] > offset:
        break_points.insert(0, offset)
    if break_points[-1] < end:
        break_points.append(end)
    sizes = [b - a for a, b in zip(break_points[:-1], break_points[1:])]
    return chunk.split(sizes)


def all_gather_into_tensor_flat_fp8(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    output_shape: torch.Size,
    group: dist.ProcessGroup,
    fp8_format: str = "e4m3",
):
    """all gather into tensor in fp8 format

    Args:
        output_tensor (torch.Tensor): output tensor, which is flattened
        input_tensor (torch.Tensor): input tensor, which is flattened
        group (dist.ProcessGroup): process group
        fp8_format (str, optional): fp8 format, e4m3 or e5m2. Defaults to "e4m3".
    """
    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "input/output tensor should be flattened"
    world_size = dist.get_world_size(group)
    assert (
        output_tensor.numel() == input_tensor.numel() * world_size
    ), "output tensor size should be world_size times of input tensor size"

    input_type = output_tensor.dtype

    fp8_type = torch.float8_e4m3fn if fp8_format == "e4m3" else torch.float8_e5m2
    fp8_max = torch.finfo(fp8_type).max

    if len(output_shape) == 2:
        per_channel_max = torch.zeros(output_shape[0], device=output_tensor.device, dtype=torch.float)
        num_channels, channel_size = output_shape
        rank = dist.get_rank(group)
        channel_start_idx = (input_tensor.numel() * rank) // channel_size
        per_channel_splits = split_chunk_by_channel(input_tensor, channel_size, num_channels, rank, world_size)
        for i, per_channel_split in enumerate(per_channel_splits):
            idx = i + channel_start_idx
            if idx < num_channels:
                per_channel_max[idx] = per_channel_split.abs().max().float()
        dist.all_reduce(per_channel_max, op=dist.ReduceOp.MAX, group=group)
        per_channel_max = torch.where(per_channel_max > 0, per_channel_max, 1.0)
        scale = fp8_max / per_channel_max
        fp8_input = input_tensor.float()
        fp8_per_channel_splits = split_chunk_by_channel(fp8_input, channel_size, num_channels, rank, world_size)
        for i, per_channel_split in enumerate(fp8_per_channel_splits):
            idx = i + channel_start_idx
            if idx < num_channels:
                per_channel_split.mul_(scale[idx])
        fp8_input = fp8_input.to(fp8_type)
    else:
        per_tensor_max = input_tensor.abs().max().float()
        dist.all_reduce(per_tensor_max, op=dist.ReduceOp.MAX, group=group)
        per_tensor_max = torch.where(per_tensor_max > 0, per_tensor_max, 1.0)
        scale = fp8_max / per_tensor_max
        fp8_input = (scale * input_tensor.float()).to(fp8_type)
    scale_inv = 1.0 / scale
    buffer = torch.empty_like(output_tensor, dtype=fp8_type)
    dist.all_gather_into_tensor(buffer.view(torch.uint8), fp8_input.view(torch.uint8), group=group)
    numel = np.prod(output_shape)
    valid_buffer = buffer[:numel].reshape(output_shape)
    valid_buffer = cast_from_fp8(valid_buffer, scale_inv, input_type)
    output_tensor[:numel].copy_(valid_buffer.view(-1))

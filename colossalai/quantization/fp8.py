from typing import Any

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


def reduce_scatter_fp8(output: torch.Tensor, input_list, group, fp8_format="e4m3") -> None:
    r"""
    This is an in-place operation for compressed all_reduce using fp8.
    It works like dist.all_reduce but during communication the data is cast to fp8 format.

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

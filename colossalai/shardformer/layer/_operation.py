import functools

import torch
import torch.distributed as dist
import torch.nn.functional as F

from colossalai.pipeline.weight_grad_store import WeightGradStore

from .utils import (
    execute_conv1d_w_pass,
    execute_conv1d_w_pass_grad_accum,
    execute_w_pass,
    execute_w_pass_grad_accum,
    is_share_sp_tp,
)

try:
    import fused_mix_prec_layer_norm_cuda
except:
    fused_mix_prec_layer_norm_cuda = None

try:
    import fused_weight_gradient_mlp_cuda

    _grad_accum_fusion_available = True
except ImportError:
    _grad_accum_fusion_available = False

from colossalai.quantization.fp8 import (
    all_gather_fp8,
    all_reduce_fp8,
    all_to_all_fp8,
    all_to_all_single_fp8,
    reduce_scatter_fp8,
)


class FusedLayerNormAffineFunction1D(torch.autograd.Function):
    r"""Layernorm

    Args:
        input: input matrix.
        weight: weight matrix.
        bias: bias matrix.
        normalized_shape: input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1] \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability
    """

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_mix_prec_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )

        return grad_input, grad_weight, grad_bias, None, None


class MatmulWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False, use_zbv=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.fp8_communication = fp8_communication
        ctx.use_zbv = use_zbv

        output = torch.matmul(input_, weight)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        fp8_communication = ctx.fp8_communication
        use_zbv = ctx.use_zbv

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias.
        weight_origin = weight
        weight = weight.view(weight.shape)
        if bias is not None:
            bias = bias.view(bias.shape)

        total_input = input
        grad_input = grad_output.matmul(weight.T)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if fp8_communication or not ctx.async_grad_allreduce:
            _reduce(grad_input, group=ctx.process_group, fp8_communication=fp8_communication, fp8_format="e5m2")
        elif ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=ctx.process_group, async_op=True)
            # Rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to have
            # all-reduce scheduled first and have GPU resources allocated, CUDA_DEVICE_MAX_CONNECTIONS=1 is set in shardformer.py

        # split dx & dw
        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    (weight, weight_origin),
                    functools.partial(
                        execute_conv1d_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = total_input.t().matmul(grad_output)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    (weight, weight_origin),
                    functools.partial(
                        execute_conv1d_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = total_input.t().matmul(grad_output)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce and not fp8_communication:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None


class MatmulWithGradAccum(torch.autograd.Function):
    """
    Linear layer execution with grad accum in backprop. (no tp version)
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, async_grad_allreduce, use_zbv=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.use_zbv = use_zbv

        output = torch.matmul(input_, weight)
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        use_zbv = ctx.use_zbv

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias.
        weight_origin = weight
        weight = weight.view(weight.shape)
        if bias is not None:
            bias = bias.view(bias.shape)

        total_input = input
        grad_input = grad_output.matmul(weight.T)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        # split dx & dw
        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad

            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    (weight, weight_origin),
                    functools.partial(
                        execute_conv1d_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = total_input.t().matmul(grad_output)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    (weight, weight_origin),
                    functools.partial(
                        execute_conv1d_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = total_input.t().matmul(grad_output)

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None


class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False, use_zbv=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.fp8_communication = fp8_communication
        ctx.use_zbv = use_zbv
        if bias is not None:
            output = F.linear(input_, weight, bias)
        else:
            output = F.linear(input_, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        fp8_communication = ctx.fp8_communication
        use_zbv = ctx.use_zbv

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to bias.
        if use_bias:
            bias.view(bias.shape)

        total_input = input.contiguous()
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            if fp8_communication:
                all_reduce_fp8(grad_input, group=ctx.process_group)
            else:
                handle = dist.all_reduce(grad_input, group=ctx.process_group, async_op=True)
            # Relay on CUDA_DEVICE_MAX_CONNECTIONS=1 to have
            # all-reduce scheduled first and have GPU resources allocated, CUDA_DEVICE_MAX_CONNECTIONS=1 is set in shardformer.py
        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = grad_output.t().matmul(total_input)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce and not fp8_communication:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


class LinearWithGradAccum(torch.autograd.Function):
    """
    Linear layer baseline (no tensor parallel version).
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, async_grad_allreduce, use_zbv=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.use_zbv = use_zbv
        if bias is not None:
            output = F.linear(input_, weight, bias)
        else:
            output = F.linear(input_, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        use_zbv = ctx.use_zbv

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to bias.
        if use_bias:
            bias.view(bias.shape)

        total_input = input.contiguous()
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = grad_output.t().matmul(total_input)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None


def _ring_as_gather(func, input_to_gather=None, input_local=None, process_group=None, gather_dim=1, keep_item=False):
    # currently only support one single tensor as output
    group_size = dist.get_world_size(process_group)
    cur_rank = dist.get_rank(process_group)

    # output_tensors = [torch.empty((input_shape[0], input_shape[1], weight_shape[0])) for _ in range(group_size)]

    # initialization of ring communication
    recv_rank = cur_rank + 1 if cur_rank + 1 < group_size else 0
    send_rank = cur_rank - 1 if cur_rank > 0 else group_size - 1
    rank_map = list(dist.get_process_group_ranks(process_group))
    recv_rank = rank_map[recv_rank]
    send_rank = rank_map[send_rank]
    recv_tensors = {}
    send_tensors = {}
    for k, v in input_to_gather.items():
        recv_tensors[k] = torch.empty_like(v)
        send_tensors[k] = v.clone()

    def communicate_step():
        comm_ops = []
        for k in recv_tensors:
            comm_ops.append(dist.P2POp(dist.irecv, recv_tensors[k], recv_rank, group=process_group))
            comm_ops.append(dist.P2POp(dist.isend, send_tensors[k], send_rank, group=process_group))
        return dist.batch_isend_irecv(comm_ops)

    def switch_step():
        for k in recv_tensors:
            send_tensors[k], recv_tensors[k] = recv_tensors[k], send_tensors[k]

    input_tensors = []
    output_tensors = []

    handles = communicate_step()
    # first round: special case, retrive from local tensor
    input_tensors.append(input_to_gather)
    output_tensors.append(func(**input_to_gather, **input_local))
    for i in range(group_size - 2):
        for handle in handles:
            handle.wait()

        switch_step()

        handles = communicate_step()

        # actual computation
        input_tensors.append(send_tensors)
        output_tensors.append(func(**send_tensors, **input_local))

    # final round: special case, no need to send/recv again
    for handle in handles:
        handle.wait()
    input_tensors.append(send_tensors)
    output_tensors.append(func(**recv_tensors, **input_local))

    gathered_input = {}
    for k in input_to_gather:
        input_shards = [d[k] for d in input_tensors[group_size - cur_rank :] + input_tensors[: group_size - cur_rank]]
        gathered_input[k] = torch.cat(input_shards, dim=gather_dim)

    gathered_output = torch.cat(
        output_tensors[group_size - cur_rank :] + output_tensors[: group_size - cur_rank], dim=gather_dim
    )

    return gathered_output, gathered_input


class _GatherForwardReduceScatterBackward(torch.autograd.Function):
    """Gather input from sequence parallel in forward and reduce-scatter gradient in backward

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.
        overlap (`bool`): Whther to overlap the all_gather op and gradient calculate in backward.

    """

    @staticmethod
    def forward(ctx, input_, process_group, dim, fp8_communication=False):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.fp8_communication = fp8_communication

        return _gather(input_, dim, process_group, fp8_communication, fp8_format="e4m3")

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        process_group = ctx.process_group
        fp8_communication = ctx.fp8_communication
        # do reduce-scatter
        new_shape = list(grad_output.shape)
        assert (
            new_shape[dim] % dist.get_world_size(process_group) == 0
        ), f"The dimension to split ({new_shape[dim]}) is not a multiple of tensor parallel size ({dist.get_world_size(process_group)}). "
        new_shape[dim] = new_shape[dim] // dist.get_world_size(process_group)
        grad_list = [
            item.contiguous() for item in torch.chunk(grad_output, dist.get_world_size(process_group), dim=dim)
        ]
        output = torch.empty(new_shape, dtype=grad_output.dtype, device=grad_output.device)

        if fp8_communication:
            reduce_scatter_fp8(output, grad_list, group=process_group, fp8_format="e5m2")
        else:
            dist.reduce_scatter(output, grad_list, group=process_group)

        return output, None, None, None


class _LinearWithGatherForwardReduceScatterBackward(torch.autograd.Function):
    """Gather input from sequence parallel in forward and reduce-scatter gradient in backward

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.
        overlap (`bool`): Whether to overlap the all_gather op and gradient calculate in backward.

    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring=False, use_zbv=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_reduce_scatter = async_grad_reduce_scatter
        ctx.dim = dim
        ctx.use_zbv = use_zbv

        if ring is True:
            input_to_gather = {"input": input_}
            input_local = {"weight": weight}

            output, input_dict = _ring_as_gather(
                F.linear,
                input_to_gather=input_to_gather,
                input_local=input_local,
                process_group=process_group,
            )
            ctx.gathered_input = input_dict["input"]

            if bias is not None:
                output += bias
        else:
            input_parallel = _gather(input_, dim, process_group)
            ctx.gathered_input = input_parallel
            if bias is not None:
                output = F.linear(input_parallel, weight, bias)
            else:
                output = F.linear(input_parallel, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        dim = ctx.dim
        process_group = ctx.process_group
        use_zbv = ctx.use_zbv

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm
        if use_bias:
            bias = bias.view(bias.shape)

        input_parallel = ctx.gathered_input

        total_input = input_parallel
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_reduce_scatter:
            # Asynchronous reduce-scatter
            input_list = [
                item.contiguous() for item in torch.chunk(grad_input, dist.get_world_size(process_group), dim=dim)
            ]
            output = torch.empty(input_.shape, dtype=input_parallel.dtype, device=input_parallel.device).contiguous()
            handle = dist.reduce_scatter(output, input_list, group=process_group, async_op=True)
            # Rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to have
            # all-reduce scheduled first and have GPU resources allocated, CUDA_DEVICE_MAX_CONNECTIONS=1 is set in shardformer.py

        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = grad_output.t().matmul(total_input)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_reduce_scatter:
            handle.wait()

        return output, grad_weight, grad_bias, None, None, None, None, None


def _ring_as_reducescatter(
    func, input_to_reducescatter=None, input_local=None, process_group=None, reducescatter_dim=1
):
    # currently only support one single tensor as output
    group_size = dist.get_world_size(process_group)
    cur_rank = dist.get_rank(process_group)

    # initialization of ring communication
    recv_rank = cur_rank - 1 if cur_rank > 0 else group_size - 1
    send_rank = cur_rank + 1 if cur_rank + 1 < group_size else 0
    rank_map = list(dist.get_process_group_ranks(process_group))
    recv_rank = rank_map[recv_rank]
    send_rank = rank_map[send_rank]
    input_tensors = []
    for _ in range(group_size):
        input_tensors.append({})
    for k, v in input_to_reducescatter.items():
        input_shape = v.shape
        assert input_shape[reducescatter_dim] % group_size == 0
        _input_tensors = list(torch.split(v, input_shape[reducescatter_dim] // group_size, dim=reducescatter_dim))
        for i in range(group_size):
            input_tensors[i][k] = _input_tensors[i]
    input_tensors = input_tensors[cur_rank:] + input_tensors[:cur_rank]
    input_tensors.reverse()

    output_tensor = func(**input_tensors[0], **input_local)
    recv_tensor = torch.empty_like(output_tensor)
    send_tensor = output_tensor.clone()

    def communicate_step():
        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
        send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
        return dist.batch_isend_irecv([recv_op, send_op])

    handles = communicate_step()
    # first round: special case, retrive from local tensor
    for i in range(group_size - 2):
        # actual computation
        output_tensor = func(**input_tensors[i + 1], **input_local)

        for handle in handles:
            handle.wait()
        output_tensor += recv_tensor

        tmp_tensor = send_tensor
        send_tensor = output_tensor
        output_tensor = tmp_tensor

        handles = communicate_step()

    # final round: special case, no need to send/recv again
    output_tensor = func(**input_tensors[-1], **input_local)
    for handle in handles:
        handle.wait()
    output_tensor += recv_tensor
    return output_tensor


class _LinearWithReduceScatterForwardGatherBackward(torch.autograd.Function):
    """Reduce-scatter input from sequence parallel in forward and gather gradient in backward with ring

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.
        overlap (`bool`): Whther to overlap the all_gather op and gradient calculate in backward.

    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, dim, ring, use_zbv=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.use_zbv = use_zbv

        if ring is True:
            input_to_reducescatter = {"input": input_}
            input_local = {"weight": weight}

            if bias is not None:
                input_to_reducescatter["bias"] = bias

            output = _ring_as_reducescatter(
                F.linear,
                input_to_reducescatter=input_to_reducescatter,
                input_local=input_local,
                process_group=process_group,
            )
        else:
            if bias is not None:
                partial_output = F.linear(input_, weight, bias)
            else:
                partial_output = F.linear(input_, weight)

            output_shape = list(partial_output.shape)
            assert (
                output_shape[dim] % dist.get_world_size(process_group) == 0
            ), f"The dimension to split ({output_shape[dim]}) is not a multiple of tensor parallel size ({dist.get_world_size(process_group)}). "
            output_shape[dim] = output_shape[dim] // dist.get_world_size(process_group)

            output_list = [
                item.contiguous() for item in torch.chunk(partial_output, dist.get_world_size(process_group), dim=dim)
            ]
            output = torch.empty(output_shape, dtype=partial_output.dtype, device=partial_output.device).contiguous()
            dist.reduce_scatter(output, output_list, group=process_group)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        dim = ctx.dim
        process_group = ctx.process_group
        use_zbv = ctx.use_zbv
        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm
        if use_bias:
            bias = bias.view(bias.shape)

        grad_output = _gather(grad_output, dim, process_group)

        # TODO Need to fully optimize
        total_input = input_
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.reshape(-1, total_input.shape[-1])

        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = grad_output.t().matmul(total_input)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    weight,
                    functools.partial(
                        execute_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)

        # grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None, None


class _ReduceScatterForwardGatherBackward(torch.autograd.Function):
    """Reduce-scatter input from sequence parallel in forward and gather gradient in backward

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.

    """

    @staticmethod
    def forward(ctx, input_, process_group, dim, fp8_communication=False):
        ctx.dim = dim
        ctx.process_group = process_group
        ctx.fp8_communication = fp8_communication

        # do reduce-scatter
        new_shape = list(input_.shape)
        assert (
            new_shape[dim] % dist.get_world_size(process_group) == 0
        ), f"The dimension to split ({new_shape[dim]}) is not a multiple of tensor parallel size ({dist.get_world_size(process_group)}). "
        new_shape[dim] = new_shape[dim] // dist.get_world_size(process_group)
        input_list = [item.contiguous() for item in torch.chunk(input_, dist.get_world_size(process_group), dim=dim)]
        output = torch.empty(new_shape, dtype=input_.dtype, device=input_.device)
        if fp8_communication:
            reduce_scatter_fp8(output, input_list, group=process_group, fp8_format="e4m3")
        else:
            dist.reduce_scatter(output, input_list, group=process_group)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        process_group = ctx.process_group
        fp8_communication = ctx.fp8_communication

        return _gather(grad_output, dim, process_group, fp8_communication, fp8_format="e5m2"), None, None, None


class _MatmulWithGatherForwardReduceScatterBackward(torch.autograd.Function):
    """
    This class is designed for matmul operation with gather forward and reduce-scatter backward.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(
        ctx, input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring, fp8_communication, use_zbv=False
    ):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_reduce_scatter = async_grad_reduce_scatter
        ctx.dim = dim
        ctx.fp8_communication = fp8_communication
        ctx.use_zbv = use_zbv

        if ring is True:
            input_to_gather = {"input": input_}
            input_local = {"other": weight}

            output, input_dict = _ring_as_gather(
                torch.matmul,
                input_to_gather=input_to_gather,
                input_local=input_local,
                process_group=process_group,
                gather_dim=dim,
            )
            ctx.gathered_input = input_dict["input"]

        else:
            input_parallel = _gather(input_, dim, process_group, fp8_communication, fp8_format="e4m3")
            ctx.gathered_input = input_parallel
            output = torch.matmul(input_parallel, weight)

        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        dim = ctx.dim
        process_group = ctx.process_group
        use_zbv = ctx.use_zbv

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm
        weight_origin = weight
        weight = weight.view(weight.shape)
        if use_bias:
            bias = bias.view(bias.shape)

        input_parallel = ctx.gathered_input

        total_input = input_parallel
        grad_input = grad_output.matmul(weight.T)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_reduce_scatter:
            # Asynchronous reduce-scatter
            input_list = [
                item.contiguous() for item in torch.chunk(grad_input, dist.get_world_size(process_group), dim=dim)
            ]
            output = torch.empty(input_.shape, dtype=input_parallel.dtype, device=input_parallel.device).contiguous()
            handle = dist.reduce_scatter(output, input_list, group=process_group, async_op=True)
            # Rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to have
            # all-reduce scheduled first and have GPU resources allocated

        # split dx & dw
        if _grad_accum_fusion_available and weight.grad is not None:
            grad = weight.grad
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    (weight, weight_origin),
                    functools.partial(
                        execute_conv1d_w_pass_grad_accum,
                    ),
                )
                grad_weight = None
            else:
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = total_input.t().matmul(grad_output)
        else:
            if use_zbv:
                WeightGradStore.put(
                    total_input,
                    grad_output,
                    (weight, weight_origin),
                    functools.partial(
                        execute_conv1d_w_pass,
                        wgrad_gemm_func=torch.matmul,
                    ),
                )
                grad_weight = None
            else:
                grad_weight = total_input.t().matmul(grad_output)

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_reduce_scatter:
            handle.wait()

        return output, grad_weight, grad_bias, None, None, None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(ctx, input_, dim, process_group, grad_scale=None, fp8_communication=False):
        ctx.process_group = process_group
        # Keep the unpadded extent so the gradient can be trimmed after the
        # equal-sized collective.  Sequence-parallel inputs are not always
        # divisible by the group size (for example, a final packed batch).
        # Normalising the dimension here also makes ``narrow`` work with
        # negative dimensions in the backward pass.
        ctx.dim = dim if dim >= 0 else input_.dim() + dim
        ctx.input_dim_size = input_.size(ctx.dim)
        ctx.grad_scale = grad_scale
        ctx.fp8_communication = fp8_communication
        return _split(input_, ctx.dim, process_group, pad_to_world_size=True)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale is not None:
            grad_output = grad_output * ctx.grad_scale

        grad_input = _gather(grad_output, ctx.dim, ctx.process_group, ctx.fp8_communication, fp8_format="e5m2")
        # The collective operates on the padded extent, but the input to this
        # autograd function did not contain those synthetic tokens.  Trim
        # before returning so padding can never contribute to a valid input
        # gradient (even when a downstream kernel produced a non-zero padded
        # gradient).
        if grad_input.size(ctx.dim) != ctx.input_dim_size:
            grad_input = grad_input.narrow(ctx.dim, 0, ctx.input_dim_size).contiguous()

        return (
            grad_input,
            None,
            None,
            None,
            None,
        )


class _ReduceForward(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        process_group: communication group.

    """

    @staticmethod
    def forward(ctx, input_, process_group, grad_scale=None, fp8_communication=False):
        ctx.grad_scale = grad_scale
        return _reduce(input_, process_group, fp8_communication, fp8_format="e4m3")

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale is not None:
            grad_output = grad_output * ctx.grad_scale
        return grad_output, None, None, None


class _ReduceBackward(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
    """

    @staticmethod
    def forward(ctx, input_, process_group, fp8_communication=False):
        ctx.process_group = process_group
        ctx.fp8_communication = fp8_communication
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        fp8_communication = ctx.fp8_communication
        return _reduce(grad_output, ctx.process_group, fp8_communication, fp8_format="e5m2"), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def forward(ctx, input_, dim, process_group, grad_scale=None, fp8_communication=False, output_dim_size=None):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.output_dim_size = output_dim_size
        ctx.input_dim_size = input_.size(dim)

        output = _gather(input_, dim, process_group, fp8_communication=fp8_communication, fp8_format="e4m3")
        if output_dim_size is not None:
            if output_dim_size < 0:
                output_dim_size += output.size(dim)
            assert output_dim_size <= output.size(dim), (
                f"The requested output extent ({output_dim_size}) cannot exceed the gathered extent "
                f"({output.size(dim)})"
            )
            if output_dim_size != output.size(dim):
                output = output.narrow(dim, 0, output_dim_size).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale is not None:
            grad_output = grad_output * ctx.grad_scale

        # A trimmed forward output may have an extent which is not divisible by
        # the process-group size.  Restore the collective extent before
        # splitting the gradient; the synthetic tail has no corresponding
        # source value and is therefore discarded by the input-side trim.
        if ctx.output_dim_size is not None:
            padded_dim_size = ctx.input_dim_size * dist.get_world_size(ctx.process_group)
            if grad_output.size(ctx.dim) != padded_dim_size:
                pad_shape = list(grad_output.shape)
                pad_shape[ctx.dim] = padded_dim_size - grad_output.size(ctx.dim)
                grad_output = torch.cat((grad_output, grad_output.new_zeros(pad_shape)), dim=ctx.dim)
            return (
                _split(grad_output, ctx.dim, ctx.process_group, pad_to_world_size=True),
                None,
                None,
                None,
                None,
                None,
            )

        return _split(grad_output, ctx.dim, ctx.process_group), None, None, None, None, None


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, fp8_communication=False):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.fp8_communication = fp8_communication
        world_size = dist.get_world_size(process_group)
        bsz = input_.shape[0]

        # using all_to_all_single when batch size is 1
        if bsz == 1:
            return _all_to_all_single(
                input_,
                world_size,
                process_group,
                scatter_dim,
                gather_dim,
                fp8_communication=fp8_communication,
                fp8_format="e4m3",
            )
        else:
            return _all_to_all(
                input_,
                world_size,
                process_group,
                scatter_dim,
                gather_dim,
                fp8_communication=fp8_communication,
                fp8_format="e4m3",
            )

    @staticmethod
    def backward(ctx, grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        fp8_communication = ctx.fp8_communication
        world_size = dist.get_world_size(process_group)
        bsz = grad_output.shape[0]

        if bsz == 1:
            return_grad = _all_to_all_single(
                grad_output,
                world_size,
                process_group,
                scatter_dim,
                gather_dim,
                fp8_communication=fp8_communication,
                fp8_format="e5m2",
            )
        else:
            return_grad = _all_to_all(
                grad_output,
                world_size,
                process_group,
                scatter_dim,
                gather_dim,
                fp8_communication=fp8_communication,
                fp8_format="e5m2",
            )

        return (return_grad, None, None, None, None)


class HookParameter(torch.autograd.Function):
    """In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm"""

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(weight, bias)
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, bias = ctx.saved_tensors
        if weight is not None:
            weight = weight.view(weight.shape)
        if bias is not None:
            bias = bias.view(bias.shape)
        return grad_output, None, None


def hook_parameter_in_backward(input, weight=None, bias=None):
    return HookParameter.apply(input, weight, bias)


def _reduce(input_, process_group, fp8_communication=False, fp8_format="e5m2"):
    # skip if only one rank involved
    if dist.get_world_size(process_group) == 1:
        return input_
    else:
        if fp8_communication:
            all_reduce_fp8(input_, group=process_group, fp8_format=fp8_format)
        else:
            dist.all_reduce(input_, group=process_group)
        return input_


def _pad_sequence_tensor(tensor, target_length, dim, value=0):
    """Pad a tensor along one sequence dimension without changing its dtype."""
    if tensor is None or tensor.size(dim) >= target_length:
        return tensor

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - tensor.size(dim)
    padding = tensor.new_zeros(pad_shape) if value == 0 else tensor.new_full(pad_shape, value)
    return torch.cat((tensor, padding), dim=dim)


def pad_sequence_parallel_inputs(hidden_states, attention_mask, position_ids, cache_position, target_length):
    """Align sequence metadata with a padded sequence-parallel hidden state.

    Equal-sized communication buffers are needed by sequence-parallel
    collectives.  This helper pads the hidden state and its positional/mask
    inputs together; callers can retain the pre-padding length to trim outputs.
    """
    hidden_states = _pad_sequence_tensor(hidden_states, target_length, dim=1)

    if attention_mask is not None:
        if attention_mask.dim() <= 2:
            attention_mask = _pad_sequence_tensor(attention_mask, target_length, dim=-1)
        else:
            mask_pad_value = torch.finfo(attention_mask.dtype).min if attention_mask.is_floating_point() else 0
            attention_mask = _pad_sequence_tensor(attention_mask, target_length, dim=-1, value=mask_pad_value)
            attention_mask = _pad_sequence_tensor(attention_mask, target_length, dim=-2, value=mask_pad_value)

    if position_ids is not None:
        position_ids = _pad_sequence_tensor(position_ids, target_length, dim=-1)

    if cache_position is not None:
        old_length = cache_position.size(-1)
        if old_length < target_length:
            if old_length == 0:
                next_position = torch.arange(target_length, device=cache_position.device, dtype=cache_position.dtype)
            else:
                next_position = cache_position[..., -1:] + torch.arange(
                    1,
                    target_length - old_length + 1,
                    device=cache_position.device,
                    dtype=cache_position.dtype,
                )
            cache_position = torch.cat((cache_position, next_position), dim=-1)

    return hidden_states, attention_mask, position_ids, cache_position


def _split(input_, dim=-1, process_group=None, pad_to_world_size=False):
    """Return the rank-local chunk, optionally padding for an even split.

    Padding is opt-in because callers other than the sequence-parallel
    autograd operation rely on the existing divisibility check.  When enabled,
    zeros are appended only to the collective buffer; the corresponding
    autograd wrapper trims the gathered gradient back to the input extent.
    """
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    if dim_size % world_size != 0:
        if not pad_to_world_size:
            raise AssertionError(
                f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
                f"cannot split tensor evenly"
            )

        padded_dim_size = ((dim_size + world_size - 1) // world_size) * world_size
        pad_shape = list(input_.shape)
        pad_shape[dim] = padded_dim_size - dim_size
        input_ = torch.cat((input_, input_.new_zeros(pad_shape)), dim=dim)
        dim_size = padded_dim_size

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = dist.get_rank(process_group)
    output = tensor_list[rank].clone().contiguous()

    return output


def _gather(input_, dim=-1, process_group=None, fp8_communication=False, fp8_format="e5m2"):
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    if fp8_communication:
        all_gather_fp8(tensor_list, input_, fp8_format=fp8_format, group=process_group)
    else:
        dist.all_gather(tensor_list, input_, group=process_group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _reduce_scatter(input_, dim=1, process_group=None):
    """Do reduce-scatter operation.

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        dim (int): The dimension to perform reduce-scatter.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.
    """
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # reduce-scatter
    new_shape = list(input_.shape)
    assert (
        new_shape[dim] % dist.get_world_size(process_group) == 0
    ), f"The dimension to split ({new_shape[dim]}) is not a multiple of tensor parallel size ({dist.get_world_size(process_group)}). "
    new_shape[dim] = new_shape[dim] // world_size
    output = torch.empty(new_shape, dtype=input_.dtype, device=input_.device)
    dist.reduce_scatter(output, input_, group=process_group)

    return output


def _all_to_all(input_, world_size, group, scatter_dim, gather_dim, fp8_communication=False, fp8_format="e5m2"):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    if fp8_communication:
        all_to_all_fp8(output_list, input_list, group=group, fp8_format=fp8_format)
    else:
        dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(
    input_, seq_world_size, group, scatter_dim, gather_dim, fp8_communication=False, fp8_format="e5m2"
):
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    if scatter_dim < 2:
        input_t = input_.reshape([seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]).contiguous()
    else:
        input_t = (
            input_.reshape([-1, seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :])
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    if fp8_communication:
        all_to_all_single_fp8(output, input_t, group=group, fp8_format=fp8_format)
    else:

        dist.all_to_all_single(output, input_t, group=group)

    if scatter_dim < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_dim]
        + [
            inp_shape[gather_dim] * seq_world_size,
        ]
        + inp_shape[gather_dim + 1 :]
    ).contiguous()


def matmul_with_async_comm(
    input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False, use_zbv=False
):
    return MatmulWithAsyncCommunication.apply(
        input_, weight, bias, process_group, async_grad_allreduce, fp8_communication, use_zbv
    )


def matmul_with_grad_comm(input_, weight, bias, async_grad_allreduce, use_zbv=False):
    return MatmulWithGradAccum.apply(input_, weight, bias, async_grad_allreduce, use_zbv)


def linear_with_async_comm(
    input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False, use_zbv=False
):
    return LinearWithAsyncCommunication.apply(
        input_, weight, bias, process_group, async_grad_allreduce, fp8_communication, use_zbv
    )


def linear_with_grad_accum(input_, weight, bias, async_grad_allreduce, use_zbv=False):
    return LinearWithGradAccum.apply(input_, weight, bias, async_grad_allreduce, use_zbv)


def linear_gather_forward_reducescatter_backward(
    input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring=False, use_zbv=False
):
    return _LinearWithGatherForwardReduceScatterBackward.apply(
        input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring, use_zbv
    )


def gather_forward_reducescatter_backward(input_, process_group, dim, fp8_communication=False):
    return _GatherForwardReduceScatterBackward.apply(input_, process_group, dim, fp8_communication)


def reducescatter_forward_gather_backward(input_, process_group, dim, fp8_communication=False):
    return _ReduceScatterForwardGatherBackward.apply(input_, process_group, dim, fp8_communication)


def linear_reducescatter_forward_gather_backward(
    input_, weight, bias=None, process_group=None, dim=1, ring=False, use_zbv=False
):
    return _LinearWithReduceScatterForwardGatherBackward.apply(input_, weight, bias, process_group, dim, ring, use_zbv)


def matmul_gather_forward_reducescatter_backward(
    input_,
    weight,
    bias,
    process_group,
    async_grad_reduce_scatter,
    dim,
    ring=False,
    fp8_communication=False,
    use_zbv=False,
):
    return _MatmulWithGatherForwardReduceScatterBackward.apply(
        input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring, fp8_communication, use_zbv
    )


def gather_forward_split_backward(
    input_, dim, process_group, grad_scale=None, fp8_communication=False, output_dim_size=None
):
    return _GatherForwardSplitBackward.apply(input_, dim, process_group, grad_scale, fp8_communication, output_dim_size)


def split_forward_gather_backward(input_, dim, process_group, grad_scale=None, fp8_communication=False):
    return _SplitForwardGatherBackward.apply(input_, dim, process_group, grad_scale, fp8_communication)


def reduce_forward(input_, process_group, grad_scale=None, fp8_communication=False):
    return _ReduceForward.apply(input_, process_group, grad_scale, fp8_communication)


def reduce_backward(input_, process_group, fp8_communication=False):
    return _ReduceBackward.apply(input_, process_group, fp8_communication)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1, fp8_communication=False):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, fp8_communication)


def gather_sp_output(hidden_states, shard_config, sp_dim=1, original_dim_size=None):
    """
    Gather the output of the last layer for cross entropy computation
    """
    sp_group = shard_config.sequence_parallel_process_group
    sp_mode = shard_config.sequence_parallelism_mode
    fp8_comm = shard_config.fp8_communication
    if dist.get_world_size(sp_group) == 1:
        return hidden_states

    # Rescale grad (HybridParallelPlugin applies ZeRO grad averaging on the DP * SP group)
    scale = None if is_share_sp_tp(sp_mode) else dist.get_world_size(sp_group)
    hidden_states = gather_forward_split_backward(
        hidden_states,
        sp_dim,
        sp_group,
        grad_scale=scale,
        fp8_communication=fp8_comm,
        output_dim_size=original_dim_size,
    )
    return hidden_states

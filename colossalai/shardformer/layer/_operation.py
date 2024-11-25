import functools

import torch
import torch.distributed as dist
import torch.nn.functional as F

from colossalai.pipeline.weight_grad_store import WeightGradStore

from .utils import is_share_sp_tp

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
    def forward(ctx, input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.fp8_communication = fp8_communication

        output = torch.matmul(input_, weight)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        fp8_communication = ctx.fp8_communication

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias.
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

        grad_weight = total_input.t().matmul(grad_output)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce and not fp8_communication:
            handle.wait()

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

        def execute_w_pass_grad_accum(_input_, _grad_output_, _weight_main_grad_, wgrad_gemm_accum_func=None):
            wgrad_gemm_accum_func(_input_, _grad_output_, _weight_main_grad_)

        def execute_w_pass(_input_, _grad_output_, _weight_main_grad_=None, wgrad_gemm_func=None):
            return wgrad_gemm_func(_grad_output_.t(), _input_)

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
                # TODO: append input, grad_output_, weight, grad func to WeightGradStore
                if grad.dtype == torch.float32:
                    WeightGradStore.put(
                        total_input,
                        grad_output,
                        weight,
                        functools.partial(
                            execute_w_pass_grad_accum,
                            wgrad_gemm_accum_func=fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32,
                        ),
                    )
                    grad_weight = None
                elif grad.dtype in (torch.float16, torch.bfloat16):
                    WeightGradStore.put(
                        total_input,
                        grad_output,
                        weight,
                        functools.partial(
                            execute_w_pass_grad_accum,
                            wgrad_gemm_accum_func=fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16,
                        ),
                    )
                    grad_weight = None
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
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

        def execute_w_pass_grad_accum(_input_, _grad_output_, _weight_main_grad_, wgrad_gemm_accum_func=None):
            wgrad_gemm_accum_func(_input_, _grad_output_, _weight_main_grad_)

        def execute_w_pass(_input_, _grad_output_, _weight_main_grad_=None, wgrad_gemm_func=None):
            return wgrad_gemm_func(_grad_output_.t(), _input_)

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
                # TODO: append input, grad_output_, weight, grad func to WeightGradStore
                if grad.dtype == torch.float32:
                    WeightGradStore.put(
                        total_input,
                        grad_output,
                        weight,
                        functools.partial(
                            execute_w_pass_grad_accum,
                            wgrad_gemm_accum_func=fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32,
                        ),
                    )
                    grad_weight = None
                elif grad.dtype in (torch.float16, torch.bfloat16):
                    WeightGradStore.put(
                        total_input,
                        grad_output,
                        weight,
                        functools.partial(
                            execute_w_pass_grad_accum,
                            wgrad_gemm_accum_func=fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16,
                        ),
                    )
                    grad_weight = None
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
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
    def forward(ctx, input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring=False):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_reduce_scatter = async_grad_reduce_scatter
        ctx.dim = dim

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
            if grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, grad)
                grad_weight = None
            elif grad.dtype == torch.float16:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, grad)
                grad_weight = None
            else:
                grad_weight = grad_output.t().matmul(total_input)
        else:
            grad_weight = grad_output.t().matmul(total_input)

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_reduce_scatter:
            handle.wait()

        return output, grad_weight, grad_bias, None, None, None, None


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
    def forward(ctx, input_, weight, bias, process_group, dim, ring):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.dim = dim

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
        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None, None


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
    def forward(ctx, input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring, fp8_communication):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_reduce_scatter = async_grad_reduce_scatter
        ctx.dim = dim
        ctx.fp8_communication = fp8_communication

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

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm
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

        grad_weight = total_input.t().matmul(grad_output)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_reduce_scatter:
            handle.wait()

        return output, grad_weight, grad_bias, None, None, None, None, None


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
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.fp8_communication = fp8_communication
        return _split(input_, dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale is not None:
            grad_output = grad_output * ctx.grad_scale

        return (
            _gather(grad_output, ctx.dim, ctx.process_group, ctx.fp8_communication, fp8_format="e5m2"),
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
    def forward(ctx, input_, dim, process_group, grad_scale=None, fp8_communication=False):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        return _gather(input_, dim, process_group, fp8_communication=fp8_communication, fp8_format="e4m3")

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale is not None:
            grad_output = grad_output * ctx.grad_scale
        return _split(grad_output, ctx.dim, ctx.process_group), None, None, None, None


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


def _split(input_, dim=-1, process_group=None):
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

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


def matmul_with_async_comm(input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False):
    return MatmulWithAsyncCommunication.apply(
        input_, weight, bias, process_group, async_grad_allreduce, fp8_communication
    )


def linear_with_async_comm(
    input_, weight, bias, process_group, async_grad_allreduce, fp8_communication=False, use_zbv=False
):
    return LinearWithAsyncCommunication.apply(
        input_, weight, bias, process_group, async_grad_allreduce, fp8_communication, use_zbv
    )


def linear_with_grad_accum(input_, weight, bias, async_grad_allreduce, use_zbv=False):
    return LinearWithGradAccum.apply(input_, weight, bias, async_grad_allreduce, use_zbv)


def linear_gather_forward_reducescatter_backward(
    input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring=False
):
    return _LinearWithGatherForwardReduceScatterBackward.apply(
        input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring
    )


def gather_forward_reducescatter_backward(input_, process_group, dim, fp8_communication=False):
    return _GatherForwardReduceScatterBackward.apply(input_, process_group, dim, fp8_communication)


def reducescatter_forward_gather_backward(input_, process_group, dim, fp8_communication=False):
    return _ReduceScatterForwardGatherBackward.apply(input_, process_group, dim, fp8_communication)


def linear_reducescatter_forward_gather_backward(input_, weight, bias=None, process_group=None, dim=1, ring=False):
    return _LinearWithReduceScatterForwardGatherBackward.apply(input_, weight, bias, process_group, dim, ring)


def matmul_gather_forward_reducescatter_backward(
    input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring=False, fp8_communication=False
):
    return _MatmulWithGatherForwardReduceScatterBackward.apply(
        input_, weight, bias, process_group, async_grad_reduce_scatter, dim, ring, fp8_communication
    )


def gather_forward_split_backward(input_, dim, process_group, grad_scale=None, fp8_communication=False):
    return _GatherForwardSplitBackward.apply(input_, dim, process_group, grad_scale, fp8_communication)


def split_forward_gather_backward(input_, dim, process_group, grad_scale=None, fp8_communication=False):
    return _SplitForwardGatherBackward.apply(input_, dim, process_group, grad_scale, fp8_communication)


def reduce_forward(input_, process_group, grad_scale=None, fp8_communication=False):
    return _ReduceForward.apply(input_, process_group, grad_scale, fp8_communication)


def reduce_backward(input_, process_group, fp8_communication=False):
    return _ReduceBackward.apply(input_, process_group, fp8_communication)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1, fp8_communication=False):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, fp8_communication)


def gather_sp_output(hidden_states, shard_config, sp_dim=1):
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
        hidden_states, sp_dim, sp_group, grad_scale=scale, fp8_communication=fp8_comm
    )
    return hidden_states

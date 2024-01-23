import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import fused_mix_prec_layer_norm_cuda
except:
    fused_mix_prec_layer_norm_cuda = None

try:
    import fused_weight_gradient_mlp_cuda

    _grad_accum_fusion_available = True
except ImportError:
    _grad_accum_fusion_available = False


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
    def forward(ctx, input_, weight, bias, process_group, async_grad_allreduce):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_allreduce = async_grad_allreduce

        output = torch.matmul(input_, weight)

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias.
        weight = weight.view(weight.shape)
        bias = bias.view(bias.shape)

        total_input = input
        grad_input = grad_output.matmul(weight.T)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=ctx.process_group, async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        grad_weight = total_input.t().matmul(grad_output)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, async_grad_allreduce):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_allreduce = async_grad_allreduce

        if bias is not None:
            output = F.linear(input_, weight, bias)
        else:
            output = F.linear(input_, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to bias.
        if use_bias:
            bias.view(bias.shape)

        total_input = input
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=ctx.process_group, async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

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

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def _AllgatherLinear(input_, weight, process_group):
    group_size = dist.get_world_size(process_group)
    cur_rank = dist.get_rank(process_group)

    input_shape = input_.shape
    weight_shape = weight.shape

    output_tensors = [torch.empty((input_shape[0], input_shape[1], weight_shape[0])) for _ in range(group_size)]

    # initialization of ring communication
    input_shape[1]
    recv_rank = cur_rank + 1 if cur_rank + 1 < group_size else 0
    send_rank = cur_rank - 1 if cur_rank > 0 else group_size - 1
    recv_tensor = input_.clone()
    send_tensor = input_.clone()

    recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
    send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
    handles = dist.batch_isend_irecv([send_op, recv_op])
    # first round: special case, retrive from local tensor
    output_tensors[0] = F.linear(input_, weight)
    for i in range(group_size - 2):
        for handle in handles:
            handle.wait()

        tmp_tensor = send_tensor
        send_tensor = recv_tensor
        recv_tensor = tmp_tensor

        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
        send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
        handles = dist.batch_isend_irecv([recv_op, send_op])

        # actual computation
        output_tensors[i + 1] = F.linear(send_tensor, weight)

    # final round: special case, no need to send/recv again
    for handle in handles:
        handle.wait()
    output_tensors[group_size - 1] = F.linear(recv_tensor, weight)
    return torch.cat(output_tensors[group_size - cur_rank :] + output_tensors[: group_size - cur_rank], dim=1)


class _GatherForwardReduceScatterBackward(torch.autograd.Function):
    """Gather input from sequence parallel in forward and reduce-scatter gradient in backward

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.
        overlap (`bool`): Whther to overlap the all_gather op and gradient calculate in backward.

    """

    @staticmethod
    def forward(ctx, input_, process_group, dim):
        ctx.process_group = process_group
        ctx.dim = dim

        return _gather(input_, dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        process_group = ctx.process_group 

        # do reduce-scatter
        new_shape = list(grad_output.shape)
        assert (
            new_shape[dim] % dist.get_world_size(process_group) == 0
        ), f"The dimension to split ({new_shape[dim]}) is not a multiple of tensor parallel size ({dist.get_world_size(process_group)}). "
        new_shape[dim] = new_shape[dim] // dist.get_world_size(process_group)
        grad_list = [item.contiguous() for item in torch.chunk(grad_output, dist.get_world_size(process_group), dim=dim)]
        output = torch.empty(new_shape, dtype=grad_output.dtype, device=grad_output.device)
        dist.reduce_scatter(output, grad_list, group=process_group)

        return output, None, None


class _LinearWithGatherForwardReduceScatterBackward(torch.autograd.Function):
    """Gather input from sequence parallel in forward and reduce-scatter gradient in backward

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.
        overlap (`bool`): Whther to overlap the all_gather op and gradient calculate in backward.

    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, async_grad_reduce_scatter, dim, overlap=True):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_reduce_scatter = async_grad_reduce_scatter
        ctx.dim = dim
        ctx.overlap = overlap

        if bias is not None:
            input_parallel = _gather(input_, dim, process_group)
            output = F.linear(input_parallel, weight, bias)
        else:
            output = _AllgatherLinear(input_, weight, process_group)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias = ctx.saved_tensors
        use_bias = ctx.use_bias
        dim = ctx.dim
        process_group = ctx.process_group
        overlap = ctx.overlap

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm
        if use_bias:
            bias = bias.view(bias.shape)

        if not overlap:
            input_parallel = _gather(input_, dim, process_group)

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
                output = torch.empty(
                    input_.shape, dtype=input_parallel.dtype, device=input_parallel.device
                ).contiguous()
                handle = dist.reduce_scatter(output, input_list, group=process_group, async_op=True)
                # Delay the start of weight gradient computation shortly (3us) to have
                # reduce-scatter scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=grad_output.device) + 1

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

        else:
            input_ = input_.contiguous()
            world_size = dist.get_world_size(process_group)
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

            # do all gather in is async way
            gather_handle = dist.all_gather(tensor_list, input_, group=process_group, async_op=True)
            # calculate gradient and prepare data asynchronously with all-gather
            # calculate
            grad_input = grad_output.matmul(weight)
            grad_output = grad_output.contiguous()
            # Convert the tensor shapes to 2D for execution compatibility
            if len(grad_output.shape) > 2:
                grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_bias = grad_output.sum(dim=0) if use_bias else None
            # prepare data
            input_list = [
                item.contiguous() for item in torch.chunk(grad_input, dist.get_world_size(process_group), dim=dim)
            ]
            output = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device).contiguous()
            # wait until all-gather finished
            gather_handle.wait()

            # do reduce-scatter in async way
            reducescatter_handle = dist.reduce_scatter(output, input_list, group=process_group, async_op=True)
            input_parallel = torch.cat(tensor_list, dim=dim).contiguous()
            # calculate gradient
            if len(input_parallel.shape) > 2:
                input_parallel = input_parallel.view(-1, input_parallel.shape[-1])

            if _grad_accum_fusion_available and weight.grad is not None:
                grad = weight.grad
                if grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(input_parallel, grad_output, grad)
                    grad_weight = None
                elif grad.dtype == torch.float16:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(input_parallel, grad_output, grad)
                    grad_weight = None
                else:
                    grad_weight = grad_output.t().matmul(input_parallel)
            else:
                grad_weight = grad_output.t().matmul(input_parallel)
            # grad_weight = grad_output.t().matmul(input_parallel)
            # wait until reduce-scatter finished
            reducescatter_handle.wait()

        return output, grad_weight, grad_bias, None, None, None, None


def _ReduceScatterLinear(input_, weight, process_group):
    group_size = dist.get_world_size(process_group)
    cur_rank = dist.get_rank(process_group)

    input_shape = input_.shape

    # initialization of ring communication
    # communicate(e.g.): 0->1->2->3
    # compute(e.g.): 3->2->1->0
    input_tensors = list(torch.split(input_, int(input_shape[1] / group_size), dim=1))
    input_tensors = input_tensors[cur_rank:] + input_tensors[:cur_rank]
    input_tensors.reverse()
    recv_rank = cur_rank - 1 if cur_rank > 0 else group_size - 1
    send_rank = cur_rank + 1 if cur_rank + 1 < group_size else 0

    # first round: special case, no reduce operation
    output_tensor = F.linear(input_tensors[0], weight)
    recv_tensor = output_tensor.clone()
    send_tensor = output_tensor.clone()
    recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
    send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
    handles = dist.batch_isend_irecv([recv_op, send_op])
    for i in range(group_size - 2):
        # actual computation
        output_tensor = F.linear(input_tensors[i + 1], weight)

        for handle in handles:
            handle.wait()
        output_tensor += recv_tensor

        tmp_tensor = send_tensor
        send_tensor = output_tensor
        output_tensor = tmp_tensor

        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=process_group)
        send_op = dist.P2POp(dist.isend, send_tensor, send_rank, group=process_group)
        handles = dist.batch_isend_irecv([recv_op, send_op])

    # final round: special case, no need to send/recv again
    output_tensor = F.linear(input_tensors[group_size - 1], weight)
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
    def forward(ctx, input_, weight, bias, process_group, dim):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.dim = dim
        if bias is not None:
            partial_output = F.linear(input_, weight, bias)
        else:
            return _ReduceScatterLinear(input_, weight, process_group)

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
            total_input = total_input.view(-1, total_input.shape[-1])
        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        return grad_input, grad_weight, grad_bias, None, None


class _ReduceScatterForwardGatherBackward(torch.autograd.Function):
    """Reduce-scatter input from sequence parallel in forward and gather gradient in backward

    Args:
        input_ (`torch.Tensor`): The input tensor from sequence parallel region.
        process_group (`torch.distributed.ProcessGroup`): The process group used for collective communication.

    """

    @staticmethod
    def forward(ctx, input_, process_group, dim):
        ctx.dim = dim
        ctx.process_group = process_group

        # do reduce-scatter
        new_shape = list(input_.shape)
        assert (
            new_shape[dim] % dist.get_world_size(process_group) == 0
        ), f"The dimension to split ({new_shape[dim]}) is not a multiple of tensor parallel size ({dist.get_world_size(process_group)}). "
        new_shape[dim] = new_shape[dim] // dist.get_world_size(process_group)
        input_list = [item.contiguous() for item in torch.chunk(input_, dist.get_world_size(process_group), dim=dim)]
        output = torch.empty(new_shape, dtype=input_.dtype, device=input_.device)
        dist.reduce_scatter(output, input_list, group=process_group)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        process_group = ctx.process_group

        return _gather(grad_output, dim, process_group), None, None


class _MatmulWithGatherForwardReduceScatterBackward(torch.autograd.Function):
    """
    This class is designed for matmul operation with gather forward and reduce-scatter backward.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(ctx, input_, weight, bias, process_group, async_grad_reduce_scatter, dim, overlap):
        ctx.save_for_backward(input_, weight, bias)
        ctx.use_bias = bias is not None
        ctx.process_group = process_group
        ctx.async_grad_reduce_scatter = async_grad_reduce_scatter
        ctx.dim = dim
        ctx.overlap = overlap

        input_parallel = _gather(input_, dim, process_group)

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
        overlap = ctx.overlap

        # In order to be hooked into Gemini's '__torch_function__', adding a view operation to weight and bias. Used in FusedLayerNorm
        weight = weight.view(weight.shape)
        if use_bias:
            bias = bias.view(bias.shape)

        if not overlap:
            input_parallel = _gather(input_, dim, process_group)

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
                output = torch.empty(
                    input_.shape, dtype=input_parallel.dtype, device=input_parallel.device
                ).contiguous()
                handle = dist.reduce_scatter(output, input_list, group=process_group, async_op=True)
                # Delay the start of weight gradient computation shortly (3us) to have
                # reduce-scatter scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=grad_output.device) + 1

            grad_weight = total_input.t().matmul(grad_output)
            grad_bias = grad_output.sum(dim=0) if use_bias else None

            if ctx.async_grad_reduce_scatter:
                handle.wait()

        else:
            world_size = dist.get_world_size(process_group)
            tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

            # do all gather in is async way
            gather_handle = dist.all_gather(tensor_list, input_, group=process_group, async_op=True)
            # calculate gradient and prepare data asynchronously with all-gather
            # calculate
            grad_input = grad_output.matmul(weight.T)
            grad_output = grad_output.contiguous()
            # Convert the tensor shapes to 2D for execution compatibility
            if len(grad_output.shape) > 2:
                grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_bias = grad_output.sum(dim=0) if use_bias else None
            # prepare data
            input_list = [
                item.contiguous() for item in torch.chunk(grad_input, dist.get_world_size(process_group), dim=dim)
            ]
            output = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device).contiguous()
            # wait until all-gather finished
            gather_handle.wait()

            # do reduce-scatter in async way
            reducescatter_handle = dist.reduce_scatter(output, input_list, group=process_group, async_op=True)
            input_parallel = torch.cat(tensor_list, dim=dim).contiguous()
            # calculate gradient
            if len(input_parallel.shape) > 2:
                input_parallel = input_parallel.view(-1, input_parallel.shape[-1])
            grad_weight = input_parallel.t().matmul(grad_output)
            # wait until reduce-scatter finished
            reducescatter_handle.wait()

        return output, grad_weight, grad_bias, None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(ctx, input_, dim, process_group):
        ctx.process_group = process_group
        ctx.dim = dim
        return _split(input_, dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.dim, ctx.process_group), None, None


class _ReduceForward(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
    """

    @staticmethod
    def forward(ctx, input_, process_group):
        return _reduce(input_, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ReduceBackward(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
    """

    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.process_group = process_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.process_group), None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def forward(ctx, input_, dim, process_group):
        ctx.process_group = process_group
        ctx.dim = dim
        return _gather(input_, dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, ctx.process_group), None, None


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)
        return _all_to_all(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


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


def hook_paramter_in_backward(input, weight=None, bias=None):
    return HookParameter.apply(input, weight, bias)


def _reduce(input_, process_group):
    # skip if only one rank involved
    if dist.get_world_size(process_group) == 1:
        return input_
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


def _gather(input_, dim=-1, process_group=None):
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # all gather
    input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=process_group)

    # concat
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


def _all_to_all(input_, world_size, group, scatter_dim, gather_dim):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def matmul_with_async_comm(input_, weight, bias, process_group, async_grad_allreduce):
    return MatmulWithAsyncCommunication.apply(input_, weight, bias, process_group, async_grad_allreduce)


def linear_with_async_comm(input_, weight, bias, process_group, async_grad_allreduce):
    return LinearWithAsyncCommunication.apply(input_, weight, bias, process_group, async_grad_allreduce)


def linear_gather_forward_reducescatter_backward(
    input_, weight, bias, process_group, async_grad_reduce_scatter, dim, overlap
):
    return _LinearWithGatherForwardReduceScatterBackward.apply(
        input_, weight, bias, process_group, async_grad_reduce_scatter, dim, overlap
    )


def gather_forward_reducescatter_backward(input_, process_group, dim):
    return _GatherForwardReduceScatterBackward.apply(input_, process_group, dim)


def reducescatter_forward_gather_backward(input_, process_group, dim):
    return _ReduceScatterForwardGatherBackward.apply(input_, process_group, dim)


def linear_reducescatter_forward_gather_backward(input_, weight, bias=None, process_group=None, dim=1):
    return _LinearWithReduceScatterForwardGatherBackward.apply(input_, weight, bias, process_group, dim)


def matmul_gather_forward_reducescatter_backward(
    input_, weight, bias, process_group, async_grad_reduce_scatter, dim, overlap
):
    return _MatmulWithGatherForwardReduceScatterBackward.apply(
        input_, weight, bias, process_group, async_grad_reduce_scatter, dim, overlap
    )


def gather_forward_split_backward(input_, dim, process_group):
    return _GatherForwardSplitBackward.apply(input_, dim, process_group)


def split_forward_gather_backward(input_, dim, process_group):
    return _SplitForwardGatherBackward.apply(input_, dim, process_group)


def reduce_forward(input_, process_group):
    return _ReduceForward.apply(input_, process_group)


def reduce_backward(input_, process_group):
    return _ReduceBackward.apply(input_, process_group)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

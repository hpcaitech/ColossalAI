import torch
import torch.distributed as dist

from colossalai.legacy.core import global_context as gpc

try:
    import fused_mix_prec_layer_norm_cuda
except:
    fused_mix_prec_layer_norm_cuda = None


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


class LinearWithAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication in backprop.
    """

    @staticmethod
    def forward(ctx, input_, weight, bias, parallel_mode, async_grad_allreduce):
        ctx.save_for_backward(input_, weight)
        ctx.use_bias = bias is not None
        ctx.parallel_mode = parallel_mode
        ctx.async_grad_allreduce = async_grad_allreduce

        output = torch.matmul(input_, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        total_input = input
        grad_input = grad_output.matmul(weight)

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = dist.all_reduce(grad_input, group=gpc.get_group(ctx.parallel_mode), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            # TODO: This seems to only work if you add torch.cuda.Event.wait()

            # _ = torch.zeros(1, device=grad_output.device)

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_async_comm(input_, weight, bias, parallel_mode, async_grad_allreduce):
    return LinearWithAsyncCommunication.apply(input_, weight, bias, parallel_mode, async_grad_allreduce)

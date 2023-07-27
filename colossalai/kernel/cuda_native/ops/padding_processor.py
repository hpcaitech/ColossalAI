import torch
from einops import rearrange


class Unpad(torch.autograd.Function):
    """
    Adapted from
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, indices: torch.Tensor):
        ctx.save_for_backward(indices)
        # [b, s, ...]
        assert tensor.ndim >= 3
        ctx.bsz = tensor.shape[0]
        out = rearrange(tensor, 'b s ... -> (b s) ...')
        ctx.shape = out.shape
        # [1, ntokens, ...]
        return out[indices].unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # [b*s, ...]
        grad = torch.zeros(ctx.shape, dtype=grad_output.dtype, device=grad_output.device)
        grad[indices] = grad_output.squeeze(0)
        grad = rearrange(grad, '(b s) ... -> b s ...', b=ctx.bsz)
        # [b, s, ...]
        return grad, None


class Repad(torch.autograd.Function):
    """
    Adapted from
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int):
        ctx.save_for_backward(indices)
        # [ntokens, ...]
        tensor = tensor.squeeze(0)
        out = torch.zeros((batch_size * seq_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        # [b*s, ...]
        out[indices] = tensor
        # [b, s, ...]
        out = rearrange(out, '(b s) ... -> b s ...', b=batch_size)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # [b*s, ...]
        grad_output = rearrange(grad_output, 'b s ... -> (b s) ...')
        grad = grad_output[indices]
        # [1, ntokens, ...]
        return grad.unsqueeze(0), None, None, None

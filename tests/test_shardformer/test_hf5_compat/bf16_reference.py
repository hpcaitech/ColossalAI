"""Independent PyTorch controls, not replacements for native-HF baselines."""

import torch
import torch.nn.functional as F


class GroupedColumn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, parts, padding_multiple):
        rows = weight.shape[0]
        multiple = parts * padding_multiple
        padded_rows = ((rows + multiple - 1) // multiple) * multiple
        padded_weight = F.pad(weight, (0, 0, 0, padded_rows - rows))
        padded_bias = None if bias is None else F.pad(bias, (0, padded_rows - rows))
        ctx.save_for_backward(inputs, padded_weight)
        ctx.has_bias = bias is not None
        ctx.rows = rows
        ctx.parts = parts
        biases = padded_bias.chunk(parts) if padded_bias is not None else [None] * parts
        output = torch.cat([F.linear(inputs, w, b) for w, b in zip(padded_weight.chunk(parts, 0), biases)], dim=-1)
        return output[..., :rows]

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_output = F.pad(grad_output, (0, weight.shape[0] - ctx.rows))
        slices = grad_output.chunk(ctx.parts, -1)
        input_gradients = [g.matmul(w) for g, w in zip(slices, weight.chunk(ctx.parts, 0))]
        grad_input = input_gradients[0]
        for partial in input_gradients[1:]:
            grad_input = grad_input + partial
        flat_input = inputs.reshape(-1, inputs.shape[-1])
        grad_weight = torch.cat([g.reshape(-1, g.shape[-1]).t().matmul(flat_input) for g in slices], dim=0)[: ctx.rows]
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)[: ctx.rows] if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None


def grouped_column(inputs, weight, bias=None, *, parts=2, padding_multiple=1):
    """Match declared physical row partitions, including zero vocabulary padding.

    This is a numerical diagnostic: it performs no distributed communication,
    uses no ColossalAI layer, and must be reported separately from native HF.
    """
    return GroupedColumn.apply(inputs, weight, bias, parts, padding_multiple)

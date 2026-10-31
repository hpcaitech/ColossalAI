import pytest
import torch
import torch.nn.functional as F

from tests.test_shardformer.test_hf5_compat.bf16_reference import grouped_column


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_physical_padded_column(dtype, bias):
    inputs = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    weight = torch.randn(13, 8, dtype=dtype, requires_grad=True)
    offset = torch.randn(13, dtype=dtype, requires_grad=True) if bias else None
    output = grouped_column(inputs, weight, offset, padding_multiple=4)
    padded_weight = F.pad(weight, (0, 0, 0, 3))
    padded_bias = F.pad(offset, (0, 3)) if offset is not None else None
    biases = padded_bias.chunk(2) if padded_bias is not None else (None, None)
    expected = torch.cat([F.linear(inputs, w, b) for w, b in zip(padded_weight.chunk(2), biases)], -1)[..., :13]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    padded_grad = F.pad(upstream, (0, 3))
    gradients = [g.matmul(w) for g, w in zip(padded_grad.chunk(2, -1), padded_weight.chunk(2))]
    torch.testing.assert_close(inputs.grad, gradients[0] + gradients[1], atol=0, rtol=0)
    expected_weight = upstream.reshape(-1, 13).t().matmul(inputs.detach().reshape(-1, 8))
    torch.testing.assert_close(weight.grad, expected_weight, atol=0.005, rtol=0.005)
    if offset is not None:
        torch.testing.assert_close(offset.grad, padded_grad.reshape(-1, 16).sum(0)[:13], atol=0, rtol=0)

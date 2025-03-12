# This file was modifed from https://github.com/deepseek-ai/DeepGEMM
# as the utils are not included in library
# Thanks for developing and open-sourcing the performant kernel

# Original LICENSE:

# MIT License

# Copyright (c) 2025 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import warnings
from typing import Tuple

import torch

__WARNING_MSG = "Couldn't find deep_gemm library, please install from https://github.com/deepseek-ai/DeepGEMM and run corresponding tests"
try:
    from deep_gemm import ceil_div, gemm_fp8_fp8_bf16_nt

    IS_DEEP_GEMM_AVAIL = True
except ImportError:
    IS_DEEP_GEMM_AVAIL = False
    warnings.warn(__WARNING_MSG)

    def ceil_dev(*args, **kwargs):  # to surpass code lint
        raise NotImplementedError(__WARNING_MSG)

    def gemm_fp8_fp8_bf16_nt(*args, **kwargs):
        raise NotImplementedError(__WARNING_MSG)


def deepgemm_fp8_gemm(
    lhs: Tuple[torch.Tensor, torch.Tensor], rhs: Tuple[torch.Tensor, torch.Tensor], out: torch.Tensor
) -> None:
    gemm_fp8_fp8_bf16_nt(lhs, rhs, out)


# TODO: There seems to be better kernel implemented in triton
@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Casting input tensor to float8_e4m3fn percicision and cooresponding scaler in token-wise mannar
    Args:
        x (`torch.Tensor`):
            Matmul x in x @ y.t(), where x.shape() is (m, k)

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`: x_float8_e4m3fn and scaler
    """
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


# TODO: There seems to be better kernel implemented in triton
@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def per_block_cast_to_fp8(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Casting input tensor to float8_e4m3fn percicision and cooresponding scaler in block-wise mannar
    Args:
        y (`torch.Tensor`):
            Matmul y in x @ y.t(), where y.shape() is (n, k)

    Returns:
        `Tuple[torch.Tensor, torch.Tensor]`: y_float8_e4m3fn and scaler
    """
    assert y.dim() == 2
    m, n = y.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=y.dtype, device=y.device)
    x_padded[:m, :n] = y
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

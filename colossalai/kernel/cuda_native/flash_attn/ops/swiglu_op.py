# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .common import BaseOperator, get_xformers_operator, register_operator
from .unbind import stack_or_none, unbind


@register_operator
class DualGemmSiluOp(BaseOperator):
    OPERATOR = get_xformers_operator("dual_gemm_silu_identity_mul")
    OPERATOR_CATEGORY = "swiglu"
    NAME = "dual_gemm_silu"

    @classmethod
    # type: ignore
    def operator_flop(
        cls, x: torch.Tensor, w1: torch.Tensor, b1, w2: torch.Tensor, b2
    ) -> int:
        """NOTE: we neglect the impact of biases / pointwises"""
        M, N, K = x.shape[0], w1.shape[0], w1.shape[1]
        return M * N * K * 2 * 2


@register_operator
class GemmFusedSumOp(BaseOperator):
    OPERATOR = get_xformers_operator("gemm_fused_operand_sum")
    OPERATOR_CATEGORY = "swiglu"
    NAME = "gemm_fused_operand_sum"

    @classmethod
    # type: ignore
    def operator_flop(cls, a: torch.Tensor, b: torch.Tensor, out1, out2) -> int:
        M, N, K = a.shape[0], b.shape[1], a.shape[1]
        return M * N * K * 2


class _SwiGLUDecomposedFunc(torch.autograd.Function):
    """
    This is just an example implementation with all
    operations explicited. This implementation is worse
    than pytorch, because pytorch is able to fuse some operations
    (eg the linear forward ...) that are decomposed here.

    The time measurements were made on the ViT-Giant setting:
    - A100/f16
    - input: [4440, 1536]
    - hidden: [4440, 4096]
    """

    NAME = "decomposed"
    FORCE_BW_F32 = False

    def _silu_backward(dy, x):
        # https://github.com/pytorch/pytorch/blob/563b065f5a4b4055fa6b025c2514b566d5fd9439/aten/src/ATen/native/Activation.cpp#L483
        sigm = 1 / (1 + torch.exp(-x.float()))
        return (dy.float() * sigm * (1 + x.float() * (1 - sigm))).to(x.dtype)

    # 952us
    @classmethod
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1 = x @ w1.transpose(-2, -1) + b1  # 275us
        x2 = x @ w2.transpose(-2, -1) + b2  # 275us
        x3 = F.silu(x1)  # 62us
        x4 = x3 * x2  # 90us
        x5 = x4 @ w3.transpose(-2, -1) + b3  # 250us

        ctx.save_for_backward(x, w1, b1, w2, b2, w3, b3, x1, x2, x3, x4, x5)
        return x5

    # 1900us
    @classmethod
    def backward(cls, ctx, dx5):
        saved_tensors = ctx.saved_tensors
        if cls.FORCE_BW_F32:
            dx5 = dx5.float()
            saved_tensors = [t.float() for t in ctx.saved_tensors]
        x, w1, b1, w2, b2, w3, b3, x1, x2, x3, x4, x5 = saved_tensors
        dx4 = dx5 @ w3  # 255us (nn)
        dw3 = dx5.transpose(-2, -1) @ x4  # 247us (nt)
        db3 = dx5.sum(0)  # 25us
        dx3 = dx4 * x2  # 88us
        dx2 = dx4 * x3  # 88us
        dx1 = cls._silu_backward(dx3, x1)  # 90us
        dx = dx2 @ w2  # 260us (nn)
        dw2 = dx2.transpose(-2, -1) @ x  # 245us (nt)
        db2 = dx2.sum(0)  # 50us
        dx += dx1 @ w1  # 260us (nn)
        dw1 = dx1.transpose(-2, -1) @ x  # 245us (nt)
        db1 = dx1.sum(0)  # 50us
        return (dx, dw1, db1, dw2, db2, dw3, db3)


class _SwiGLUFusedFunc(torch.autograd.Function):
    NAME = "fused.py"

    @classmethod
    @torch.cuda.amp.custom_fwd
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1, x2, x4 = DualGemmSiluOp.OPERATOR(x, w1, b1, w2, b2)

        x5 = F.linear(x4, w3, b3)
        ctx.save_for_backward(x, w1, w2, w3, x1, x2)
        ctx.bias = [b1 is not None, b2 is not None, b3 is not None]
        return x5

    @staticmethod
    def _linear_bw(
        dy: torch.Tensor, x: torch.Tensor, bias: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not bias:
            return (dy.transpose(-2, -1) @ x), None
        db = torch.empty([dy.shape[1]], dtype=dy.dtype, device=dy.device)
        dw = torch.empty([dy.shape[1], x.shape[1]], dtype=dy.dtype, device=dy.device)
        GemmFusedSumOp.OPERATOR(dy.transpose(-2, -1), x, dw, db)
        return dw, db

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, dx5):
        x, w1, w2, w3, x1, x2 = ctx.saved_tensors
        w1w2 = stack_or_none([w1, w2], dim=0)

        dx4 = dx5 @ w3  # 255us (nn)
        dx1dx2, x4 = torch.ops.xformers.silu_bw_fused(x1, x2, dx4)
        dx1, dx2 = dx1dx2.unbind(1)
        del x1, x2, dx4

        dw3, db3 = cls._linear_bw(dx5, x4, bias=ctx.bias[2])
        del x4, dx5
        if w1w2 is not None:
            assert dx1dx2.is_contiguous()
            assert w1w2.is_contiguous()
            w1w2 = w1w2.view([w1.shape[0] * 2, w1.shape[1]])
            dx = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]) @ w1w2

            # backward of linear1 + linear2 - packed
            dw1dw2 = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]).transpose(-2, -1) @ x
            dw1dw2, db1db2 = cls._linear_bw(
                dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]), x, bias=ctx.bias[0]
            )
            dw1, dw2 = dw1dw2.view([2, *w1.shape]).unbind(0)
            if ctx.bias[0]:
                db1db2 = db1db2.view([2, dx1.shape[1]])
                db1, db2 = torch.unbind(db1db2, dim=0)
            else:
                db1 = db2 = None
        else:
            dx = dx2 @ w2  # 260us (nn)
            torch.addmm(
                dx, dx1, w1.to(dx1.dtype), beta=1, alpha=1, out=dx
            )  # dx += dx1 @ w1
            dw2, db2 = cls._linear_bw(dx2, x, bias=ctx.bias[1])
            dw1, db1 = cls._linear_bw(dx1, x, bias=ctx.bias[0])
        return (dx, dw1, db1, dw2, db2, dw3, db3)


class SwiGLUOp:
    """Base class for any swiglu operator in :attr:`xformers.ops.swiglu`"""

    def __init__(self, op, packed_weights: bool, name: str, constraints):
        self.NAME = name
        self.PACKED_WEIGHTS = packed_weights
        self.op = op
        self.constraints = constraints

    def supports(self, op: "SwiGLUOpDispatch") -> bool:
        if self.PACKED_WEIGHTS and not op.packed_weights:
            return False
        return all(c(op) for c in self.constraints)

    def __call__(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        pass

    def __str__(self) -> str:
        return f"SwiGLUOp:{self.NAME}"


class _ForwardToPythonAutogradFunc(SwiGLUOp):
    def supports(self, op: "SwiGLUOpDispatch") -> bool:
        # Let's disable autocast in bf16 until this issue is fixed
        # https://github.com/pytorch/pytorch/issues/87979
        if op.dtype_autocast_gpu == torch.bfloat16:
            return False
        return super().supports(op)

    def __call__(self, *args, **kwargs):
        return self.op.apply(*args, **kwargs)


class _ForwardToFunc(SwiGLUOp):
    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)

    def info(self):
        if self.op.__name__ == "no_such_operator":
            return "not built"
        return "available"


def _eager_functional_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    x1 = F.linear(x, w1, b1)
    x2 = F.linear(x, w2, b2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3, b3)


@dataclass
class SwiGLUOpDispatch:
    """Dispatcher to automatically select
    the best operator in :attr:`xformers.ops.swiglu`
    """

    device: Union[torch.device, str]
    dtype: torch.dtype
    dtype_autocast_gpu: Optional[torch.dtype]
    packed_weights: bool
    bias_enabled: bool

    @property
    def op(self) -> SwiGLUOp:
        """Computes the best operator

        Returns:
            SwiGLUOp: The best operator for the configuration
        """
        priorities: Sequence[SwiGLUOp] = [
            SwiGLUPackedFusedOp,
            SwiGLUFusedOp,
        ]
        for op in priorities:
            if op.supports(self):
                return op
        return SwiGLUEagerOp

    @staticmethod
    def from_arguments(
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: Optional[torch.Tensor],
        w2: torch.Tensor,
        b2: Optional[torch.Tensor],
        w3: torch.Tensor,
        b3: Optional[torch.Tensor],
    ) -> "SwiGLUOpDispatch":
        return SwiGLUOpDispatch(
            device=x.device,
            dtype=x.dtype,
            packed_weights=stack_or_none((w1, w2), dim=0) is not None,
            dtype_autocast_gpu=torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else w1.dtype,
            bias_enabled=b1 is not None and b2 is not None and b3 is not None,
        )


def _only_sm80(op: SwiGLUOpDispatch) -> bool:
    device_type = op.device if isinstance(op.device, str) else op.device.type
    return device_type == "cuda" and torch.cuda.get_device_capability(op.device)[0] >= 8


def _only_half_or_autocast(op: SwiGLUOpDispatch) -> bool:
    HALF_DTYPES = [torch.half, torch.bfloat16]
    return op.dtype in HALF_DTYPES or (
        op.dtype_autocast_gpu is not None and op.dtype_autocast_gpu in HALF_DTYPES
    )


def _bias_enabled(op: SwiGLUOpDispatch) -> bool:
    return op.bias_enabled


_SwiGLUDecomposedOp = _ForwardToPythonAutogradFunc(
    _SwiGLUDecomposedFunc, False, "decomposed", constraints=[_bias_enabled]
)
SwiGLUFusedOp = _ForwardToPythonAutogradFunc(
    _SwiGLUFusedFunc, False, "fused", constraints=[_only_sm80, _only_half_or_autocast]
)
SwiGLUPackedFusedOp = _ForwardToFunc(
    get_xformers_operator("swiglu_packedw"),
    True,
    "fused.p.cpp",
    constraints=[_only_sm80, _only_half_or_autocast],
)
SwiGLUEagerOp = _ForwardToFunc(
    _eager_functional_swiglu,
    False,
    "eager",
    constraints=[],
)


def _info() -> Dict[str, str]:
    return {op.NAME: op.info() for op in [SwiGLUPackedFusedOp]}


def swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    b3: Optional[torch.Tensor],
    *,
    op: SwiGLUOp = None,
) -> torch.Tensor:
    """
    Computes a SwiGLU block given the weights/bias of the 3
    linear layers.

    - It is recommended to keep ``op=None`` so the best implementation \
    available for the inputs will be used.


    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, w3, b3)

    :Packing weights:

    To allow faster implementations, it's recommended to have w1/w2 come from the same storage, as in:
        .. code-block:: python

            w1, w2 = xformers.ops.unbind(w12, 0)

    :Supported hardware:

    This operator is only optimized on A100+ on ``torch.half`` or ``torch.bfloat16`` \
        (autocast is supported), and will fallback to a functional pytorch \
        implementation otherwise.
    """

    batch_shape = x.shape[:-1]
    x = x.reshape([-1, x.shape[-1]])
    if w1.ndim != 2 or w1.shape != w2.shape:
        raise ValueError(f"Invalid shapes for w1: {w1.shape} / w2: {w2.shape}")
    if b1 is not None:
        if b1.ndim != 1 or b1.shape[0] != w1.shape[0]:
            raise ValueError(f"Invalid shapes for b1: {b1.shape}")
    if b2 is not None:
        if b2.ndim != 1 or b2.shape[0] != w2.shape[0]:
            raise ValueError(f"Invalid shapes for b2: {b2.shape}")
    if w3.ndim != 2 or w3.shape[1] != w2.shape[0]:
        raise ValueError(f"Invalid shape for w3: {w3.shape}")
    if b3 is not None:
        if b3.ndim != 1 or b3.shape[0] != w3.shape[0]:
            raise ValueError(f"Invalid shapes for w3: {w3.shape} / b3: {b3.shape}")

    if op is None:
        op = SwiGLUOpDispatch.from_arguments(x, w1, b1, w2, b2, w3, b3).op

    if not op.PACKED_WEIGHTS:
        return op(x, w1, b1, w2, b2, w3, b3).reshape([*batch_shape, -1])
    w1w2 = stack_or_none((w1, w2), dim=0)
    if b1 is not None and b2 is not None:
        b1b2: Optional[torch.Tensor] = stack_or_none((b1, b2), dim=0)
        if b1b2 is None:
            raise NotImplementedError("b1/b2 needs to be properly packed")
    else:
        b1b2 = None
        assert b1 is None and b2 is None

    if w1w2 is None:
        raise NotImplementedError("w1/w2 needs to be properly packed")
    return op(x, w1w2, b1b2, w3, b3).reshape([*batch_shape, -1])


class SwiGLU(nn.Module):
    """
    A Module that encapsulates the call to :attr:`xformers.ops.swiglu`,
    and holds the weights for the 3 linear layers
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        *,
        _pack_weights: bool = True,
    ) -> None:
        """Create a SwiGLU module

        Args:
            in_features (int): Number of features of the input
            hidden_features (int): Number of hidden features
            out_features (Optional[int], optional): Number of features of the input. Defaults to None.
            bias (bool, optional): Whether linear layers also include a bias. Defaults to True.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12: Optional[nn.Linear]
        if _pack_weights:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features
        self.op = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes :attr:`swiglu` with the module's weights

        Args:
            x (torch.Tensor): A Tensor of shape ``[..., in_features]``

        Returns:
            torch.Tensor: A Tensor of shape ``[..., out_features]``
        """
        return swiglu(x, *self._ordered_params(), op=self.op)

    def _ordered_params(self):
        """Used for testing - returns ordered arguments for operators"""
        b1: Optional[torch.Tensor]
        b2: Optional[torch.Tensor]
        if self.w12 is not None:
            w1w2 = self.w12.weight
            b1b2 = self.w12.bias
            w1, w2 = unbind(
                w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]),
                dim=0,
            )
            if b1b2 is not None:
                b1, b2 = unbind(b1b2.view([2, b1b2.shape[0] // 2]), dim=0)
            else:
                b1, b2 = None, None
        else:
            w1, w2 = self.w1.weight, self.w2.weight
            b1, b2 = self.w1.bias, self.w2.bias
        return [
            w1,
            b1,
            w2,
            b2,
            self.w3.weight,
            self.w3.bias,
        ]

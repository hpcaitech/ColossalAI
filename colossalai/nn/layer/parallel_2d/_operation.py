from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from colossalai.communication.collective import (all_gather, all_reduce, reduce, reduce_scatter)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from colossalai.global_variables import tensor_parallel_env as env


def matmul_2d(
    a,
    b,
    summa_dim,
    out_shape,
    row_rank=None,
    col_rank=None,
    row_parallel_mode=ParallelMode.PARALLEL_2D_ROW,
    col_parallel_mode=ParallelMode.PARALLEL_2D_COL,
):
    r"""Matrix multiplication for 2D parallelism.

    Args:
        a (:class:`torch.tensor`): matrix :math:`A`.
        b (:class:`torch.tensor`): matrix :math:`B`.
        summa_dim (int): dimension of SUMMA fo 2D parallelism.
        out_shape (:class:`torch.size`): shape of output tensor.
        row_rank (int, optional): the rank of row, defaults to None.
        col_rank (int, optional): the rank of column, defaults to None.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`, optional):
            row parallel mode, defaults to ParallelMode.PARALLEL_2D_ROW.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`, optional):
            column parallel mode, defaults to ParallelMode.PARALLEL_2D_COL.

    Returns:
        :class:`torch.tensor`: :math:`C = AB`.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    if row_rank is None:
        row_rank = gpc.get_local_rank(col_parallel_mode)
    if col_rank is None:
        col_rank = gpc.get_local_rank(row_parallel_mode)

    data_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.DATA) else gpc.get_local_rank(ParallelMode.DATA)
    pipeline_parallel_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(
        ParallelMode.PIPELINE)
    pipeline_parallel_size = 1 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_world_size(
        ParallelMode.PIPELINE)
    tensor_parallel_size = summa_dim**2
    return Matmul_AB_2D(a, b, summa_dim, out_shape, row_rank, col_rank, row_parallel_mode, col_parallel_mode,
                        data_parallel_rank, pipeline_parallel_rank, pipeline_parallel_size, tensor_parallel_size)


class _Classifier2D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        bias: Optional[Tensor],
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        A = A.clone().detach()
        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        B_temp = all_gather(B, -1, col_parallel_mode)
        if ctx:
            ctx.save_for_backward(A, B_temp)

        C = torch.matmul(A, B_temp.transpose(0, 1))

        C = all_reduce(C, row_parallel_mode)

        ctx.use_bias = bias is not None
        if bias is not None:
            C = C + bias

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = torch.matmul(output_grad, B)
            A_grad = A_grad.reshape(ctx.A_shape)
            B_grad = torch.matmul(output_grad.reshape(-1, output_grad.shape[-1]).transpose(0, 1), A)
            B_grad = reduce_scatter(B_grad, -1, ctx.col_parallel_mode)
            B_grad = B_grad.reshape(ctx.B_shape)
            if ctx.use_bias:
                bias_grad = torch.sum(output_grad, dim=tuple(range(output_grad.ndim - 1)))
                bias_grad = all_reduce(bias_grad, ctx.col_parallel_mode)
            else:
                bias_grad = None

        return A_grad, B_grad, bias_grad, None, None, None, None, None, None, None, None, None, None


def classifier_2d(A: Tensor, B: Tensor, bias: Optional[Tensor], summa_dim: int, out_shape: Tuple[int, ...],
                  row_rank: int, col_rank: int, row_parallel_mode: ParallelMode, col_parallel_mode: ParallelMode,
                  data_parallel_rank: int, pipeline_parallel_rank: int, pipeline_parallel_size: int,
                  tensor_parallel_size: int) -> Tensor:
    r"""2D parallel classifier.

    Args:
        A (:class:`torch.tensor`): matrix :math:`A`.
        B (:class:`torch.tensor`): matrix :math:`B`.
        bias (:class:`torch.tensor`, optional): matrix of bias.
        summa_dim (int): dimension of SUMMA fo 2D parallelism.
        out_shape (:class:`torch.size`): shape of output tensor.
        row_rank (int, optional): the rank of row, defaults to None.
        col_rank (int, optional): the rank of column, defaults to None.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`): row parallel mode.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`): column parallel mode.
        data_parallel_rank (int): data parallel rank.
        pipeline_parallel_rank (int): pipeline parallel rank
        pipeline_parallel_size (int): pipeline parallel size.
        tensor_parallel_size (int): tensor parallel size.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Classifier2D.apply(A, B, bias, summa_dim, out_shape, row_rank, col_rank, row_parallel_mode,
                               col_parallel_mode, data_parallel_rank, pipeline_parallel_rank, pipeline_parallel_size,
                               tensor_parallel_size)


class Matmul_AB_2D(torch.autograd.Function):
    r"""Matrix multiplication for :math:`C = AB`.

    Args:
        A (:class:`torch.tensor`): matrix :math:`A`.
        B (:class:`torch.tensor`): matrix :math:`B`.
        summa_dim (int): dimension of SUMMA fo 2D parallelism.
        out_shape (:class:`torch.size`): shape of output tensor.
        row_rank (int, optional): the rank of row, defaults to None.
        col_rank (int, optional): the rank of column, defaults to None.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`): row parallel mode.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`): column parallel mode.
        data_parallel_rank (int): data parallel rank.
        pipeline_parallel_rank (int): pipeline parallel rank
        pipeline_parallel_size (int): pipeline parallel size.
        tensor_parallel_size (int): tensor parallel size.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        # A: [b / q, s, h / q] -> [(b * s) / q, h / q]
        # B: [h / q, s / q]
        # C: [b / q, s, s / q] -> [(b * s) / q, s / q]

        assert A.shape[-1] == B.shape[-2], \
            'Invalid shapes: A={}, B={} for AB.'.format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[-1])
        C = torch.zeros(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        B_list = [torch.empty_like(B) for _ in range(2)]

        row_group = gpc.get_group(row_parallel_mode)
        col_group = gpc.get_group(col_parallel_mode)

        src_a = summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
        src_b = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size

        opa = [None] * 2
        opb = [None] * 2

        A_list[0].copy_(A)
        B_list[0].copy_(B)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True)
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(B_list[1 - cur], src=src_b + summa_dim, group=col_group, async_op=True)

            if opa[cur] is not None:
                opa[cur].wait()
            if opb[cur] is not None:
                opb[cur].wait()

            torch.addmm(C, A_list[cur], B_list[cur], out=C)
            cur = 1 - cur
            src_a += 1
            src_b += summa_dim

        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors
        with torch.no_grad():
            A_grad = Matmul_ABT_2D.apply(output_grad, B, ctx.summa_dim, ctx.A_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
            B_grad = Matmul_ATB_2D.apply(A, output_grad, ctx.summa_dim, ctx.B_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_ABT_2D(torch.autograd.Function):
    r"""Matrix multiplication for :math:`C = AB^T`

    Args:
        A (:class:`torch.tensor`): matrix :math:`A`.
        B (:class:`torch.tensor`): matrix :math:`B`.
        summa_dim (int): dimension of SUMMA fo 2D parallelism.
        out_shape (:class:`torch.size`): shape of output tensor.
        row_rank (int, optional): the rank of row, defaults to None.
        col_rank (int, optional): the rank of column, defaults to None.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`): row parallel mode.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`): column parallel mode.
            column parallel mode, defaults to ParallelMode.PARALLEL_2D_COL.
        data_parallel_rank (int): data parallel rank.
        pipeline_parallel_rank (int): pipeline parallel rank
        pipeline_parallel_size (int): pipeline parallel size.
        tensor_parallel_size (int): tensor parallel size.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        assert A.shape[-1] == B.shape[-1], \
            'Invalid shapes: A={}, B={} for ABT.'.format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[0], B.shape[0])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        B_list = [torch.empty_like(B) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = gpc.get_group(row_parallel_mode)
        col_group = gpc.get_group(col_parallel_mode)

        src_b = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
        src_c = summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size

        opb = [None] * 2
        opr = [None] * 2

        B_list[0].copy_(B)
        opb[0] = dist.broadcast(B_list[0], src=src_b, group=col_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                B_list[1 - cur].copy_(B)
                opb[1 - cur] = dist.broadcast(B_list[1 - cur], src=src_b + summa_dim, group=col_group, async_op=True)

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == col_rank:
                    C.copy_(C_list[cur])

            if opb[cur] is not None:
                opb[cur].wait()

            torch.matmul(A, B_list[cur].transpose(0, 1), out=C_list[cur])
            opr[cur] = dist.reduce(C_list[cur], dst=src_c, group=row_group, async_op=True)
            cur = 1 - cur
            src_b += summa_dim
            src_c += 1

        for op in opr:
            op.wait()

        if summa_dim - 2 == col_rank:
            C.copy_(C_list[cur])
        if summa_dim - 1 == col_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = Matmul_AB_2D.apply(output_grad, B, ctx.summa_dim, ctx.A_shape, ctx.row_rank, ctx.col_rank,
                                        ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                        ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                        ctx.tensor_parallel_size)
            B_grad = Matmul_ATB_2D.apply(output_grad, A, ctx.summa_dim, ctx.B_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class Matmul_ATB_2D(torch.autograd.Function):
    r"""Matrix multiplication for :math:`C = A^TB`.

    Args:
        A (:class:`torch.tensor`): matrix :math:`A`.
        B (:class:`torch.tensor`): matrix :math:`B`.
        summa_dim (int): dimension of SUMMA fo 2D parallelism.
        out_shape (:class:`torch.size`): shape of output tensor.
        row_rank (int, optional): the rank of row, defaults to None.
        col_rank (int, optional): the rank of column, defaults to None.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`): row parallel mode.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`): column parallel mode.
        data_parallel_rank (int): data parallel rank.
        pipeline_parallel_rank (int): pipeline parallel rank
        pipeline_parallel_size (int): pipeline parallel size.
        tensor_parallel_size (int): tensor parallel size.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        A: Tensor,
        B: Tensor,
        summa_dim: int,
        out_shape: Tuple[int, ...],
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:

        assert A.shape[-2] == B.shape[-2], \
            'Invalid shapes: A={}, B={} for ATB.'.format(A.shape, B.shape)

        if ctx:
            ctx.save_for_backward(A, B)

        A_shape = A.shape
        A = A.reshape((-1, A_shape[-1]))
        B_shape = B.shape
        B = B.reshape((-1, B_shape[-1]))
        C_shape = (A.shape[-1], B.shape[-1])
        C = torch.empty(C_shape, dtype=A.dtype, device=get_current_device())

        # use circular buffer to store the communication tensor
        # 2 is enough for all cases
        A_list = [torch.empty_like(A) for _ in range(2)]
        C_list = [torch.empty_like(C) for _ in range(2)]

        row_group = gpc.get_group(row_parallel_mode)
        col_group = gpc.get_group(col_parallel_mode)

        src_a = summa_dim * row_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size
        src_c = col_rank + data_parallel_rank * pipeline_parallel_size * tensor_parallel_size + \
                pipeline_parallel_rank * tensor_parallel_size

        opa = [None] * 2
        opr = [None] * 2

        A_list[0].copy_(A)
        opa[0] = dist.broadcast(A_list[0], src=src_a, group=row_group, async_op=True)
        cur = 0

        for i in range(summa_dim):
            if i != summa_dim - 1:
                A_list[1 - cur].copy_(A)
                opa[1 - cur] = dist.broadcast(A_list[1 - cur], src=src_a + 1, group=row_group, async_op=True)

            if opr[cur] is not None:
                opr[cur].wait()
                if i - 2 == row_rank:
                    C.copy_(C_list[cur])

            if opa[cur] is not None:
                opa[cur].wait()

            torch.matmul(A_list[cur].transpose(0, 1), B, out=C_list[cur])
            opr[cur] = dist.reduce(C_list[cur], dst=src_c, group=col_group, async_op=True)
            cur = 1 - cur
            src_a += 1
            src_c += summa_dim

        for op in opr:
            op.wait()

        if summa_dim - 2 == row_rank:
            C.copy_(C_list[cur])
        if summa_dim - 1 == row_rank:
            C.copy_(C_list[1 - cur])
        out = C.reshape(out_shape)

        if ctx:
            ctx.summa_dim = summa_dim
            ctx.row_rank = row_rank
            ctx.col_rank = col_rank
            ctx.row_parallel_mode = row_parallel_mode
            ctx.col_parallel_mode = col_parallel_mode
            ctx.A_shape = A_shape
            ctx.B_shape = B_shape
            ctx.data_parallel_rank = data_parallel_rank
            ctx.pipeline_parallel_rank = pipeline_parallel_rank
            ctx.pipeline_parallel_size = pipeline_parallel_size
            ctx.tensor_parallel_size = tensor_parallel_size

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        A, B = ctx.saved_tensors

        with torch.no_grad():
            A_grad = Matmul_ABT_2D.apply(B, output_grad, ctx.summa_dim, ctx.A_shape, ctx.row_rank, ctx.col_rank,
                                         ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                         ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                         ctx.tensor_parallel_size)
            B_grad = Matmul_AB_2D.apply(A, output_grad, ctx.summa_dim, ctx.B_shape, ctx.row_rank, ctx.col_rank,
                                        ctx.row_parallel_mode, ctx.col_parallel_mode, ctx.data_parallel_rank,
                                        ctx.pipeline_parallel_rank, ctx.pipeline_parallel_size,
                                        ctx.tensor_parallel_size)
        return A_grad, B_grad, None, None, None, None, None, None, None, None, None, None


class _Add_Bias_2D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any,
        input_: Tensor,
        bias: Tensor,
        output_size_per_partition: int,
        row_rank: int,
        col_rank: int,
        row_parallel_mode: ParallelMode,
        col_parallel_mode: ParallelMode,
        skip_bias_add: bool,
        data_parallel_rank: int,
        pipeline_parallel_rank: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
    ) -> Tensor:
        bias_temp = all_gather(bias, -1, col_parallel_mode)

        ctx.row_rank = row_rank
        ctx.col_rank = col_rank
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        ctx.bias = skip_bias_add
        ctx.data_parallel_rank = data_parallel_rank
        ctx.pipeline_parallel_rank = pipeline_parallel_rank
        ctx.pipeline_parallel_size = pipeline_parallel_size
        ctx.tensor_parallel_size = tensor_parallel_size

        if skip_bias_add:
            return bias_temp
        else:
            output = input_ + bias_temp
            return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        col_parallel_mode = ctx.col_parallel_mode

        if ctx.bias:
            grad = reduce_scatter(output_grad, -1, col_parallel_mode)
            return None, grad, None, None, None, None, None, None, None, None, None, None
        else:
            reduce_dim = tuple(range(output_grad.ndim - 1))
            reduce = torch.sum(output_grad, dim=reduce_dim)
            grad = reduce_scatter(reduce, -1, col_parallel_mode)
            return output_grad, grad, None, None, None, None, None, None, None, None, None, None


def add_bias_2d(input_: Tensor, bias: Tensor, output_size_per_partition: int, row_rank: int, col_rank: int,
                row_parallel_mode: ParallelMode, col_parallel_mode: ParallelMode, skip_bias_add: bool,
                data_parallel_rank: int, pipeline_parallel_rank: int, pipeline_parallel_size: int,
                tensor_parallel_size: int) -> Tensor:
    r"""Matrix add bias: :math:`C = A + b`.

    Args:
        input_ (:class:`torch.tensor`): matrix :math:`A`.
        bias (:class:`torch.tensor`): matrix :math:`B`.
        output_size_per_partition (int): size of output per partition.
        row_rank (int, optional): the rank of row, defaults to None.
        col_rank (int, optional): the rank of column, defaults to None.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`): row parallel mode.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`): column parallel mode.
        skip_bias_add (bool):
            If set to ``True``, it will skip bias add for linear layer, which is preserved for kernel fusion.
        data_parallel_rank (int): data parallel rank.
        pipeline_parallel_rank (int): pipeline parallel rank
        pipeline_parallel_size (int): pipeline parallel size.
        tensor_parallel_size (int): tensor parallel size.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Add_Bias_2D.apply(input_, bias, output_size_per_partition, row_rank, col_rank, row_parallel_mode,
                              col_parallel_mode, skip_bias_add, data_parallel_rank, pipeline_parallel_rank,
                              pipeline_parallel_size, tensor_parallel_size)


class _Layernorm_2D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx: Any, input_: Tensor, E_x: Tensor, Var_x: Tensor, hidden_size: int, row_parallel_mode: ParallelMode,
                col_parallel_mode: ParallelMode) -> Tensor:
        input_ = input_ - E_x
        # in here, input = x - E[x], Var_x = 1 / sqrt(Var[x] + eps)
        ctx.normalized_shape = hidden_size
        output = input_ * Var_x
        ctx.save_for_backward(output, Var_x)
        ctx.row_parallel_mode = row_parallel_mode
        ctx.col_parallel_mode = col_parallel_mode
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        row_parallel_mode = ctx.row_parallel_mode
        col_parallel_mode = ctx.col_parallel_mode
        x, Var_x = ctx.saved_tensors
        # in here, Var_x = 1 / sqrt(Var[x] + eps), x = (x - E[x]) * Var_x
        output_grad_sum = torch.sum(output_grad, dim=-1, keepdim=True)
        torch.distributed.all_reduce(output_grad_sum, group=gpc.get_group(row_parallel_mode))
        output_grad_sum /= ctx.normalized_shape

        output_grad_mul_x_sum = torch.sum(output_grad * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(output_grad_mul_x_sum, group=gpc.get_group(row_parallel_mode))
        output_grad_mul_x_sum /= ctx.normalized_shape

        input_grad = output_grad.clone()
        input_grad -= x * output_grad_mul_x_sum
        input_grad -= output_grad_sum
        input_grad *= Var_x

        return input_grad, None, None, None, None, None


def layernorm_2d(input_: Tensor, E_x: Tensor, Var_x: Tensor, hidden_size: int, row_parallel_mode: ParallelMode,
                 col_parallel_mode: ParallelMode) -> Tensor:
    r"""Layernorm.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        E_x (:class:`torch.tensor`): mean.
        Var_x (:class:`torch.tensor`): variance.
        hidden_size (int): hidden size.
        row_parallel_mode (:class:`colossalai.context.ParallelMode`): row parallel mode.
        col_parallel_mode (:class:`colossalai.context.ParallelMode`): column parallel mode.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _Layernorm_2D.apply(input_, E_x, Var_x, hidden_size, row_parallel_mode, col_parallel_mode)


class _AllGatherTensor2D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, inputs: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode

        outputs = all_gather(inputs, dim, parallel_mode)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, output_grad: Tensor) -> Tuple[Tensor, ...]:
        grad = reduce_scatter(output_grad, ctx.dim, ctx.parallel_mode)
        return grad.contiguous(), None, None


def all_gather_tensor_2d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    r"""All gather the tensor of 2D parallelism.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to gather.
        parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode tensor used.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _AllGatherTensor2D.apply(tensor, dim, parallel_mode)


def split_batch_2d(input_: Tensor, dim: int = 0) -> Tensor:
    """Splits 2D tensor in specified dimension across cols.

    Args:
        input_ (:class:`torch.tensor`): Input tensor.
        dim (int): Specified dimension in which to split.

    Returns:
        :class:`torch.tensor`: The tensor has been split.
    """
    dim_size = input_.size(dim)
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_2D_COL)

    if world_size <= 1:
        return input_

    assert dim_size % world_size == 0, \
        f'The batch size ({dim_size}) is not a multiple of 2D size ({world_size}).'

    return torch.chunk(input_, gpc.get_world_size(ParallelMode.PARALLEL_2D_COL),
                       dim=dim)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)].contiguous()


class _ReduceTensor2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, parallel_mode):
        return all_reduce(input_, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad, None


def reduce_tensor_2d(input_: Tensor, parallel_mode: ParallelMode) -> Tensor:
    r"""All-reduce the input.

    Args:
        input_ (:class:`torch.tensor`): Input tensor.
        parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode tensor used.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    return _ReduceTensor2D.apply(input_, parallel_mode)


class _ReduceScatterTensor2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, parallel_mode):
        ctx.dim = dim
        ctx.parallel_mode = parallel_mode
        return reduce_scatter(input_, dim, parallel_mode)

    @staticmethod
    def backward(ctx, output_grad):
        return all_gather(output_grad, ctx.dim, ctx.parallel_mode), None, None


def reduce_scatter_tensor_2d(tensor: Tensor, dim: int, parallel_mode: ParallelMode) -> Tensor:
    r"""Reduce-scatter the input.

    Args:
        tensor (:class:`torch.tensor`): Input tensor.
        dim (int): Dimension to reduce.
        parallel_mode (:class:`colossalai.context.ParallelMode`): The parallel mode tensor used.

    Note:
        The parallel_mode should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_
    """
    dim_size = tensor.size(dim)
    world_size = gpc.get_world_size(parallel_mode)
    assert dim_size % world_size == 0, \
        f'The batch size ({dim_size}) is not a multiple of 2D size ({world_size}).'

    return _ReduceScatterTensor2D.apply(tensor, dim, parallel_mode)


class _ReduceByBatch2D(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_, reduce_mean: bool = False):
        output = all_reduce(input_, ParallelMode.PARALLEL_2D_COL)
        if reduce_mean:
            reduce_size = gpc.get_world_size(ParallelMode.PARALLEL_2D_COL)
            return output / reduce_size
        return output

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_, reduce_mean: bool = False):
        output = all_reduce(input_, ParallelMode.PARALLEL_2D_COL)
        ctx.reduce_mean = reduce_mean
        if reduce_mean:
            reduce_size = gpc.get_world_size(ParallelMode.PARALLEL_2D_COL)
            ctx.reduce_size = reduce_size
            return output.clone() / reduce_size
        return output.clone()

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        if ctx.reduce_mean:
            return output_grad / ctx.reduce_size, None
        else:
            return output_grad, None


def reduce_by_batch_2d(input_, reduce_mean: bool = False) -> Tensor:
    r"""All-reduce the input from the model parallel region.

    Args:
        input_ (:class:`torch.tensor`): input matrix.
        reduce_mean (bool, optional):
            If set to ``True``, it will divide the output by column parallel size, default to False.
    """
    return _ReduceByBatch2D.apply(input_, reduce_mean)

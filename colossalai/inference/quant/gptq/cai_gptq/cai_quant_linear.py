# Adapted from AutoGPTQ auto_gptq: https://github.com/PanQiWei/AutoGPTQ

import math
import warnings
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import ParallelModule

from .gptq_op import CaiGPTQLinearOp

HAS_GPTQ_CUDA = False
try:
    from colossalai.kernel.op_builder.gptq import GPTQBuilder

    gptq_cuda = GPTQBuilder().load()
    HAS_GPTQ_CUDA = True
except ImportError:
    warnings.warn("CUDA gptq is not installed")
    HAS_GPTQ_CUDA = False


class CaiQuantLinear(nn.Module):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, tp_size=1, tp_rank=0, row_split=False):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer("qweight", torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer(
            "qzeros",
            torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32),
        )
        self.register_buffer(
            "scales", torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16)
        )
        if row_split:
            self.register_buffer(
                "g_idx",
                torch.tensor(
                    [(i + (tp_rank * self.infeatures)) // self.groupsize for i in range(infeatures)], dtype=torch.int32
                ),
            )
        else:
            self.register_buffer(
                "g_idx", torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32)
            )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.gptq_linear = CaiGPTQLinearOp(groupsize, bits)

        self.q4 = None
        self.empty_tensor = torch.empty((1, 1), device="meta")
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.row_split = row_split

    def pack(self, linear, scales, zeros, g_idx=None):
        g_idx = (
            g_idx.clone()
            if g_idx is not None
            else torch.tensor([i // self.groupsize for i in range(self.infeatures)], dtype=torch.int32)
        )

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        half_scales = scales.clone().half()
        # print("scale shape ", scales.shape, scale_zeros.shape, linear.weight.shape)
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        pbits = 32
        ptype = torch.int32
        unsign_type = np.uint32
        sign_type = np.int32

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round((linear.weight.data[:, idx] + scale_zeros[g_idx[idx]]) / half_scales[g_idx[idx]]).to(ptype)[
                    :, None
                ]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(unsign_type)
        qweight = np.zeros((intweight.shape[0] // pbits * self.bits, intweight.shape[1]), dtype=unsign_type)

        i = 0
        row = 0

        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (pbits // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += pbits // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qweight = qweight.astype(sign_type)
        qweight1 = torch.from_numpy(qweight)
        qweight1 = qweight1.contiguous()  # .to("cuda")
        self.qweight.data.copy_(qweight1)

        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // pbits * self.bits), dtype=unsign_type)
        zeros -= 1
        zeros = zeros.numpy().astype(unsign_type)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (pbits // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += pbits // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qzeros = qzeros.astype(sign_type)
        qzeros = torch.from_numpy(qzeros)
        qzeros = qzeros
        self.qzeros.data.copy_(qzeros)

        if torch.equal(self.g_idx.to(g_idx.device), g_idx):
            self.g_idx = None
        else:
            self.g_idx = g_idx

    def init_q4(self):
        assert self.qweight.device.type == "cuda"
        self.q4_width = self.qweight.shape[1]
        if self.g_idx is not None:
            if self.row_split and torch.equal(
                self.g_idx,
                torch.tensor(
                    [(i + (self.tp_rank * self.infeatures)) // self.groupsize for i in range(self.infeatures)],
                    dtype=torch.int32,
                    device=self.g_idx.device,
                ),
            ):
                self.g_idx = None
            elif torch.equal(
                self.g_idx,
                torch.tensor(
                    [i // self.groupsize for i in range(self.infeatures)], dtype=torch.int32, device=self.g_idx.device
                ),
            ):
                self.g_idx = None

        if self.g_idx is not None:
            g_idx = self.g_idx.to("cpu")
        else:
            g_idx = self.empty_tensor

        self.q4 = gptq_cuda.make_q4(self.qweight, self.qzeros, self.scales, g_idx, torch.cuda.current_device())
        torch.cuda.synchronize()

    def forward(self, x):
        outshape = x.shape[:-1] + (self.outfeatures,)

        if HAS_GPTQ_CUDA and self.bits == 4:
            if self.q4 is None:
                self.init_q4()

            x = x.view(-1, x.shape[-1])
            output = torch.empty((x.shape[0], self.outfeatures), dtype=torch.float16, device=x.device)
            gptq_cuda.q4_matmul(x.half(), self.q4, output)
            if self.bias is not None and (not self.row_split or self.tp_size == 1):
                output.add_(self.bias)
        else:
            if self.bias is not None and (not self.row_split or self.tp_size == 1):
                bias = self.bias
            else:
                bias = None
            output = self.gptq_linear(
                x,
                self.qweight,
                self.scales,
                self.qzeros,
                g_idx=self.g_idx,
                bias=bias,
            )
        return output.view(outshape)


def split_column_copy(gptq_linear, cai_linear, tp_size=1, tp_rank=0, split_num=1):
    qweights = gptq_linear.qweight.split(gptq_linear.out_features // split_num, dim=-1)
    qzeros = gptq_linear.qzeros.split(gptq_linear.out_features // (32 // cai_linear.bits) // split_num, dim=-1)
    scales = gptq_linear.scales.split(gptq_linear.out_features // split_num, dim=-1)
    g_idx = gptq_linear.g_idx
    if gptq_linear.bias is not None:
        bias = gptq_linear.bias.split(gptq_linear.out_features // split_num, dim=-1)

    cai_split_out_features = cai_linear.outfeatures // split_num
    zero_split_block = cai_linear.outfeatures // (32 // cai_linear.bits) // split_num

    for i in range(split_num):
        cai_linear.qweight[:, i * cai_split_out_features : (i + 1) * cai_split_out_features] = qweights[i][
            :, tp_rank * cai_split_out_features : (tp_rank + 1) * cai_split_out_features
        ]
        cai_linear.qzeros[:, i * zero_split_block : (i + 1) * zero_split_block] = qzeros[i][
            :, tp_rank * zero_split_block : (tp_rank + 1) * zero_split_block
        ]
        cai_linear.scales[:, i * cai_split_out_features : (i + 1) * cai_split_out_features] = scales[i][
            :, tp_rank * cai_split_out_features : (tp_rank + 1) * cai_split_out_features
        ]
        if cai_linear.bias is not None:
            cai_linear.bias[i * cai_split_out_features : (i + 1) * cai_split_out_features] = bias[i][
                tp_rank * cai_split_out_features : (tp_rank + 1) * cai_split_out_features
            ]

    cai_linear.g_idx.copy_(g_idx)


def split_row_copy(gptq_linear, cai_linear, tp_rank=0, split_num=1):
    qweights = gptq_linear.qweight.split(gptq_linear.in_features // split_num, dim=0)
    qzeros = gptq_linear.qzeros.split(gptq_linear.in_features // split_num, dim=0)
    scales = gptq_linear.scales.split(gptq_linear.in_features // split_num, dim=0)
    g_idxs = gptq_linear.g_idx.split(gptq_linear.in_features // split_num, dim=0)

    cai_split_in_features = cai_linear.infeatures // (32 // cai_linear.bits) // split_num
    zero_split_block = cai_linear.infeatures // cai_linear.groupsize // split_num
    idx_split_features = cai_linear.infeatures // split_num

    for i in range(split_num):
        cai_linear.qweight[i * cai_split_in_features : (i + 1) * cai_split_in_features, :] = qweights[i][
            tp_rank * cai_split_in_features : (tp_rank + 1) * cai_split_in_features, :
        ]
        cai_linear.qzeros[i * zero_split_block : (i + 1) * zero_split_block, :] = qzeros[i][
            tp_rank * zero_split_block : (tp_rank + 1) * zero_split_block, :
        ]
        cai_linear.scales[i * zero_split_block : (i + 1) * zero_split_block, :] = scales[i][
            tp_rank * zero_split_block : (tp_rank + 1) * zero_split_block, :
        ]
        cai_linear.g_idx[i * idx_split_features : (i + 1) * idx_split_features] = g_idxs[i][
            tp_rank * idx_split_features : (tp_rank + 1) * idx_split_features
        ]
    if cai_linear.bias is not None:
        cai_linear.bias.copy_(gptq_linear.bias)


class RowCaiQuantLinear(CaiQuantLinear, ParallelModule):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, tp_size=1, tp_rank=0, row_split=False):
        super().__init__(
            bits, groupsize, infeatures, outfeatures, bias, tp_size=tp_size, tp_rank=tp_rank, row_split=row_split
        )
        self.process_group = None

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.in_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if in_features < tp_size:
            return module

        if in_features % tp_size != 0:
            raise ValueError(
                f"The size of in_features:{in_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = RowCaiQuantLinear(
            module.bits,
            module.group_size,
            module.in_features // tp_size,
            module.out_features,
            module.bias is not None,
            tp_size=tp_size,
            tp_rank=tp_rank,
            row_split=True,
        )
        linear_1d.process_group = process_group

        split_row_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        return linear_1d

    def forward(self, x):
        output = super().forward(x)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
            if self.bias is not None:
                output.add_(self.bias)
        return output


class ColCaiQuantLinear(CaiQuantLinear, ParallelModule):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, tp_size=1, tp_rank=0, row_split=False):
        super().__init__(
            bits, groupsize, infeatures, outfeatures, bias, tp_size=tp_size, tp_rank=tp_rank, row_split=row_split
        )
        self.process_group = None

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.in_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if in_features < tp_size:
            return module

        if in_features % tp_size != 0:
            raise ValueError(
                f"The size of in_features:{in_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = ColCaiQuantLinear(
            module.bits,
            module.group_size,
            module.in_features,
            module.out_features // tp_size,
            module.bias is not None,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        linear_1d.process_group = process_group

        split_column_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        return linear_1d

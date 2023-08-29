# Adapted from AutoGPTQ auto_gptq: https://github.com/PanQiWei/AutoGPTQ

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import triton

from .gptq_op import CaiGPTQLinearOp

HAS_GPTQ_CUDA = False
try:
    from colossalai.kernel.op_builder.gptq import GPTQBuilder
    gptq_cuda = GPTQBuilder().load()
    HAS_GPTQ_CUDA = True
except ImportError:
    warnings.warn('CUDA gptq is not installed')
    HAS_GPTQ_CUDA = False


class CaiQuantLinear(nn.Module):
    max_dq_buffer_size = 1
    max_inner_outer_dim = 1
    max_input_len = 1
    prepared_buffers = False
    device_to_buffers = {
        "temp_state": None,
        "temp_dq": None,
    }

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, tp_size=1, tp_rank=0, row_split=False):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales',
                             torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        if row_split:
            self.register_buffer(
                'g_idx',
                torch.tensor([(i + (tp_rank * self.infeatures)) // self.groupsize for i in range(infeatures)],
                             dtype=torch.int32))
        else:
            self.register_buffer('g_idx',
                                 torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.gptq_linear = CaiGPTQLinearOp(groupsize, bits)

        self.q4 = None
        self.empty_tensor = torch.empty((1, 1), device="meta")
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.row_split = row_split

    def pack(self, linear, scales, zeros, g_idx=None):

        g_idx = g_idx.clone() if g_idx is not None else torch.tensor(
            [i // self.groupsize for i in range(self.infeatures)], dtype=torch.int32)

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        half_scales = scales.clone().half()
        # print("scale shape ", scales.shape, scale_zeros.shape, linear.weight.shape)
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        # wn = 16
        # pbits = 64
        # ptype = torch.int64
        # unsign_type = np.uint64
        # sign_type = np.int64

        wn = 8
        pbits = 32
        ptype = torch.int32
        unsign_type = np.uint32
        sign_type = np.int32

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[g_idx[idx]]) / half_scales[g_idx[idx]]).to(ptype)[:,
                                                                                                                None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(unsign_type)
        qweight = np.zeros((intweight.shape[0] // pbits * self.bits, intweight.shape[1]), dtype=unsign_type)

        i = 0
        row = 0
        # print("weight shape ", intweight.shape, qweight.shape, out_qweight.shape, bits)
        # print("weight shape ", intweight[0].shape, qweight[0].shape, out_qweight[0].shape)
        # print("weight value ", intweight[0], qweight[0])

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
        qweight1 = qweight1.contiguous()    #.to("cuda")
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

    def prepare_buffers(self):
        assert self.qweight.device.type == "cuda"
        device = self.qweight.device
        print(self.g_idx)
        if self.g_idx is not None:
            if self.row_split and torch.equal(
                    self.g_idx,
                    torch.tensor(
                        [(i + (self.tp_rank * self.infeatures)) // self.groupsize for i in range(self.infeatures)],
                        dtype=torch.int32,
                        device=self.g_idx.device)):
                self.g_idx = None
            elif torch.equal(
                    self.g_idx,
                    torch.tensor([i // self.groupsize for i in range(self.infeatures)],
                                 dtype=torch.int32,
                                 device=self.g_idx.device)):
                self.g_idx = None

        CaiQuantLinear.max_dq_buffer_size = max(CaiQuantLinear.max_dq_buffer_size, self.qweight.numel() * 8)

        if self.g_idx is not None:
            CaiQuantLinear.max_inner_outer_dim = max(CaiQuantLinear.max_inner_outer_dim, self.infeatures,
                                                     self.outfeatures)
            CaiQuantLinear.max_input_len = 4096
        # The temp_state buffer is required to reorder X in the act-order case.
        # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
        CaiQuantLinear.device_to_buffers['temp_state'] = torch.zeros(
            (CaiQuantLinear.max_input_len, CaiQuantLinear.max_inner_outer_dim), dtype=torch.float16, device=device)
        CaiQuantLinear.device_to_buffers['temp_dp'] = torch.zeros((1, CaiQuantLinear.max_dq_buffer_size),
                                                                  dtype=torch.float16,
                                                                  device=device)

        gptq_cuda.prepare_buffers(torch.device(device), CaiQuantLinear.device_to_buffers['temp_state'],
                                  CaiQuantLinear.device_to_buffers['temp_dp'])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        gptq_cuda.set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        torch.cuda.empty_cache()

    def init_q4(self):
        assert self.qweight.device.type == "cuda"
        self.q4_width = self.qweight.shape[1]
        if self.g_idx is not None:
            if self.row_split and torch.equal(
                    self.g_idx,
                    torch.tensor(
                        [(i + (self.tp_rank * self.infeatures)) // self.groupsize for i in range(self.infeatures)],
                        dtype=torch.int32,
                        device=self.g_idx.device)):
                self.g_idx = None
            elif torch.equal(
                    self.g_idx,
                    torch.tensor([i // self.groupsize for i in range(self.infeatures)],
                                 dtype=torch.int32,
                                 device=self.g_idx.device)):
                self.g_idx = None

        if self.g_idx is not None:
            g_idx = self.g_idx.to("cpu")
        else:
            g_idx = self.empty_tensor

        self.q4 = gptq_cuda.make_q4(self.qweight, self.qzeros, self.scales, g_idx, torch.cuda.current_device())
        torch.cuda.synchronize()

    def forward(self, x):
        outshape = x.shape[:-1] + (self.outfeatures,)

        if HAS_GPTQ_CUDA:
            if CaiQuantLinear.prepared_buffers == False:
                self.prepare_buffers()
                CaiQuantLinear.prepared_buffers = True

            if self.q4 is None:
                self.init_q4()

            x = x.view(-1, x.shape[-1])
            output = torch.empty((x.shape[0], self.outfeatures), dtype=torch.float16, device=x.device)
            gptq_cuda.q4_matmul(x, self.q4, output)
            if (self.bias is not None and not self.row_split) or self.tp_size == 1:
                output.add_(self.bias)
        else:
            if (self.bias is not None and not self.row_split) or self.tp_size == 1:
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


def make_cai_quant_linear(module, names, bits, groupsize, name=''):
    if isinstance(module, CaiQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr,
                    CaiQuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_cai_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)

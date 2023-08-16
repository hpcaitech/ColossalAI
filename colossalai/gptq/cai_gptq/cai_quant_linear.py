
import math
import numpy as np
import torch
import torch.nn as nn
from .gptq_op import CaiGPTQLinearOp
import triton

class CaiQuantLinear(nn.Module):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer('qweight', torch.zeros((infeatures // 64 * self.bits, outfeatures), dtype=torch.int64))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 64 * self.bits), dtype=torch.int64))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.gptq_linear = CaiGPTQLinearOp(groupsize, bits)


    def pack(self, linear, scales, zeros, g_idx=None):

        g_idx = g_idx.clone() if g_idx is not None else torch.tensor([i // self.groupsize for i in range(self.infeatures)], dtype=torch.int32)

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        half_scales = scales.clone().half()
        # print("scale shape ", scales.shape, scale_zeros.shape, linear.weight.shape)
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        wn = 16
        pbits = 64
        ptype = torch.int64
        unsign_type = np.uint64
        sign_type = np.int64

        # wn = 8
        # pbits = 32
        # ptype = torch.int32
        # unsign_type = np.uint32
        # sign_type = np.int32

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[g_idx[idx]]) / half_scales[g_idx[idx]]).to(ptype)[:, None])
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
                for j in range(i, i + (pbits //  self.bits)):
                    qweight[row] |= intweight[j] << ( self.bits * (j - i))
                i += pbits //  self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qweight = qweight.astype(sign_type)
        qweight1 = torch.from_numpy(qweight)
        qweight1 = qweight1.contiguous() #.to("cuda")
        self.qweight.data.copy_(qweight1)

        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // pbits *  self.bits), dtype=unsign_type)
        zeros -= 1
        zeros = zeros.numpy().astype(unsign_type)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if  self.bits in [2, 4, 8]:
                for j in range(i, i + (pbits //  self.bits)):
                    qzeros[:, col] |= zeros[:, j] << ( self.bits * (j - i))
                i += pbits //  self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qzeros = qzeros.astype(sign_type)
        qzeros = torch.from_numpy(qzeros)
        qzeros = qzeros
        self.qzeros.data.copy_(qzeros)
        
        if torch.equal(self.g_idx, g_idx):
            self.g_idx = None
        else:
            self.g_idx = g_idx


    def forward(self, x):

        cai_out = self.gptq_linear(x,
                            self.qweight,
                            self.scales,
                            self.qzeros,
                            g_idx = self.g_idx,
                            bias = self.bias,)
        return cai_out

def make_cai_quant_linear(module, names, bits, groupsize, name=''):
    if isinstance(module, CaiQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, CaiQuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_cai_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)


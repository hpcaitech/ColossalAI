# copied from https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/past/quant.py

import math

import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print("CUDA extension not installed.")

# Assumes layer is perfectly divisible into 256 * 256 blocks


class QuantLinear(nn.Module):
    def __init__(self, bits, groupsize, infeatures, outfeatures):
        super().__init__()
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        if groupsize != -1 and groupsize < 32 and groupsize != int(math.pow(2, int(math.log2(groupsize)))):
            raise NotImplementedError("groupsize supports powers of 2 greater than 32. (e.g. : 32,64,128,etc)")
        groupsize = groupsize if groupsize != -1 else infeatures
        self.groupsize = groupsize
        self.register_buffer(
            "qzeros", torch.zeros((math.ceil(infeatures / groupsize), outfeatures // 256 * (bits * 8)), dtype=torch.int)
        )
        self.register_buffer("scales", torch.zeros((math.ceil(infeatures / groupsize), outfeatures)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.register_buffer("qweight", torch.zeros((infeatures // 256 * (bits * 8), outfeatures), dtype=torch.int))
        self._initialized_quant_state = False

    def pack(self, linear, scales, zeros):
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.groupsize
            intweight.append(
                torch.round((linear.weight.data[:, idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[
                    :, None
                ]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros((intweight.shape[0] // 256 * (self.bits * 8), intweight.shape[1]), dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 256 * (self.bits * 8)), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        intermediate_dtype = torch.float32

        if not self._initialized_quant_state:
            # Do we even have a bias? Check for at least one non-zero element.
            if self.bias is not None and bool(torch.any(self.bias != 0)):
                # Then make sure it's the right type.
                self.bias.data = self.bias.data.to(intermediate_dtype)
            else:
                self.bias = None

        outshape = list(x.shape)
        outshape[-1] = self.outfeatures
        x = x.reshape(-1, x.shape[-1])
        if self.bias is None:
            y = torch.zeros(x.shape[0], outshape[-1], dtype=intermediate_dtype, device=x.device)
        else:
            y = self.bias.clone().repeat(x.shape[0], 1)

        output_dtype = x.dtype
        x = x.to(intermediate_dtype)
        if self.bits == 2:
            quant_cuda.vecquant2matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        elif self.bits == 3:
            quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        elif self.bits == 4:
            quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        elif self.bits == 8:
            quant_cuda.vecquant8matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(output_dtype)
        return y.reshape(outshape)


def make_quant(module, names, bits, groupsize, name=""):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + "." + name1 if name != "" else name1)

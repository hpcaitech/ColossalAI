import torch
import torch.nn as nn
import pytest
import time
import transformers
from auto_gptq.quantization import GPTQ
from auto_gptq.modeling._utils import find_layers, pack_model
from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear

from auto_gptq.quantization.quantizer import Quantizer
from colossalai.gptq import CaiGPTQLinearOp
import math
import numpy as np


wbits=4
trits=False
nsamples=1
percdamp=.01
groupsize=128
act_order=False
sym=False
class MLinear(nn.Module):
    def __init__(self, infeature, outfeature):
        super(MLinear, self).__init__()
        self.linear = torch.nn.Linear(infeature, outfeature, dtype=torch.float16)
    def forward(self, x):
        out = self.linear(x)
        return out
    
@torch.no_grad()
def model_quant(model, inps, dev):
    print('Starting ...')
    layers = [model]
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    cache = {'i': 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    # for batch in inps:
    try:
        model(inps.to(dev))
    except ValueError:
        pass
    layers[0] = layers[0].module

    outs = torch.zeros(inps.shape[0], layers[0].linear.weight.shape[0])

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)	
        gptq = {}	
        for name in subset:	
            gptq[name] = GPTQ(subset[name])	
            # gptq[name].quantizer = Quantizer()	
            gptq[name].quantizer.configure(wbits, perchannel=True, sym=sym, mse=False, trits=trits)	
            
        def add_batch(name):	
            def tmp(_, inp, out):	
                gptq[name].add_batch(inp[0].data, out.data)	
            return tmp	
            
        handles = []	
        for name in subset:	
            handles.append(subset[name].register_forward_hook(add_batch(name)))	
           
        for j in range(nsamples):	
            outs[j] = layer(inps[j].unsqueeze(0))[0]	
            
        for h in handles:	
            h.remove()	
        for name in subset:	
            print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
            scale,zero,g_idx = gptq[name].fasterquant(percdamp=percdamp, group_size=groupsize, actorder=act_order)
            # quantizers['%s' % (name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
            quantizers['%s' % (name)] = (gptq[name].layer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())

            gptq[name].free()
        for j in range(nsamples):
            layer = layer.to(dev)
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
    return quantizers


def model_pack(model, quantizers, wbits, groupsize):
    pack_model(model, quantizers, wbits, groupsize)
    return model


def cai_linear_pack(linear, scales, zeros, 
                    out_qweight, out_qscales, out_qzeros, qg_idx, 
                    infeatures, groupsize, bits):
    g_idx = qg_idx.clone() if qg_idx is not None else torch.tensor([i // groupsize for i in range(infeatures)], dtype=torch.int32)

    scales = scales.t().contiguous()
    zeros = zeros.t().contiguous()
    scale_zeros = zeros * scales
    half_scales = scales.clone().half()
    # print("scale shape ", scales.shape, scale_zeros.shape, linear.weight.shape)

    out_qscales.data.copy_(scales)

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
    for idx in range(infeatures):
        intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[g_idx[idx]]) / half_scales[g_idx[idx]]).to(ptype)[:, None])
    intweight = torch.cat(intweight, dim=1)
    intweight = intweight.t().contiguous()
    intweight = intweight.numpy().astype(unsign_type)
    qweight = np.zeros((intweight.shape[0] // pbits * bits, intweight.shape[1]), dtype=unsign_type)

    i = 0
    row = 0
    # print("weight shape ", intweight.shape, qweight.shape, out_qweight.shape, bits)
    # print("weight shape ", intweight[0].shape, qweight[0].shape, out_qweight[0].shape)
    # print("weight value ", intweight[0], qweight[0])

    while row < qweight.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (pbits // bits)):
                qweight[row] |= intweight[j] << (bits * (j - i))
            i += pbits // bits
            row += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
    qweight = qweight.astype(sign_type)
    qweight1 = torch.from_numpy(qweight)
    qweight1 = qweight1.contiguous().to("cuda")
    out_qweight.data.copy_(qweight1)

    qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // pbits * bits), dtype=unsign_type)
    zeros -= 1
    zeros = zeros.numpy().astype(unsign_type)
    i = 0
    col = 0
    while col < qzeros.shape[1]:
        if bits in [2, 4, 8]:
            for j in range(i, i + (pbits // bits)):
                qzeros[:, col] |= zeros[:, j] << (bits * (j - i))
            i += pbits // bits
            col += 1
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
    qzeros = qzeros.astype(sign_type)
    qzeros = torch.from_numpy(qzeros)
    qzeros = qzeros.to("cuda")
    out_qzeros.data.copy_(qzeros)

    return out_qweight, out_qscales, out_qzeros

def model_cai_pack(model, quantizers, qweight, qscales, qzeros, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    with torch.no_grad():
        for name in layers:
            _, scale, zero, g_idx = quantizers[name]
            qweight, qscales, qzeros = cai_linear_pack(layers[name], scale, zero, 
                        qweight, qscales, qzeros, g_idx, 
                        layers[name].weight.shape[-1], groupsize, wbits)

    # print("cai pack", layers)
    return qweight, qscales, qzeros 


def test_gptq_linear():

    infeature = 5120
    outfeature = 5120

    weight = torch.randn(outfeature, infeature).to(torch.float16).to(torch.cuda.current_device())
    bias = torch.zeros(outfeature).to(torch.float16).to(torch.cuda.current_device())
    wn = 16
    ptype = torch.int64
    
    # wn = 8
    # ptype = torch.int32

    qweight = torch.zeros(infeature//wn, outfeature, dtype=ptype, device=torch.cuda.current_device()).contiguous()
    qscales = torch.zeros(infeature//groupsize, outfeature, dtype=torch.float16, device=torch.cuda.current_device()).contiguous()
    qzeros = torch.zeros(infeature//groupsize, outfeature//wn, dtype=ptype, device=torch.cuda.current_device()).contiguous()

    act_func = nn.SiLU()
    inps = torch.ones(1, 1, infeature).to(torch.float16).to(torch.cuda.current_device())
    batch_inps = torch.randn(1, 4096, infeature).to(torch.float16).to(torch.cuda.current_device())

    linear = MLinear(infeature, outfeature)
    linear.to(torch.cuda.current_device())

    with torch.no_grad():
        linear.linear.weight.data.copy_(weight)
        linear.linear.bias.data.copy_(bias)

    with torch.no_grad():
        torch_out = linear(inps)
        batch_torch_out = linear(batch_inps)
        torch_out = act_func(torch_out)
        batch_torch_out = act_func(batch_torch_out)


    # linear.to("cuda")
    quantizers = model_quant(linear, inps, torch.cuda.current_device())
    qweight, qscales, qzeros = model_cai_pack(linear, quantizers, qweight, qscales, qzeros, wbits, groupsize)
    gptq_model = model_pack(linear, quantizers, wbits, groupsize)
    gptq_model.to(torch.cuda.current_device())
    # gptq_model = linear


    cai_linear = CaiGPTQLinearOp(groupsize, wbits)


    # qweight = torch.cat((qweight, qweight, qweight), dim=0).contiguous()
    # qscales = torch.cat((qscales, qscales, qscales), dim=0).contiguous()
    # qzeros = torch.cat((qzeros, qzeros, qzeros), dim=0).contiguous()
    # bias = torch.cat((bias, bias, bias), dim=0).contiguous()
    qkv_fused=False

    with torch.no_grad():
        gptq_out = gptq_model(inps)
        batch_gptq_out = gptq_model(batch_inps)
        cai_out = cai_linear(inps,
                            qweight,
                            qscales,
                            qzeros,
                            bias = bias, 
                            act_type = 3,
                            qkv_fused=qkv_fused)
        torch.cuda.synchronize()

        batch_cai_out = cai_linear(batch_inps,
                            qweight,
                            qscales,
                            qzeros,
                            bias=bias,
                            act_type = 3,
                            qkv_fused=qkv_fused)
        torch.cuda.synchronize()
        batch_gptq_out = act_func(batch_gptq_out)
        gptq_out = act_func(gptq_out)

    # cai_out = cai_out[1]
    # batch_cai_out = batch_cai_out[1]
    # a = torch.sum(qscales, 0)
    # print("qscales ", a)
    # print("orch out ", torch_out)
    # print("gptq out ", gptq_out)
    # print("cai out ", cai_out)
    # # print("batch_torch out ", batch_torch_out)

    # print("batch_torch out ", batch_torch_out)
    # print("batch_gptq out ", batch_gptq_out)
    # print("batch_cai out ", batch_cai_out)

    assert torch.allclose(cai_out, gptq_out, rtol=1e-01, atol=1e-02)
    assert torch.allclose(batch_cai_out, batch_gptq_out, rtol=1e-01, atol=1e-02)


    # mean_diff = torch.mean(torch.abs(cai_out - gptq_out))
    # max_diff = torch.max(torch.abs(cai_out - gptq_out))
    # print("cai vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
    # mean_diff = torch.mean(torch.abs(torch_out - gptq_out))
    # max_diff = torch.max(torch.abs(torch_out - gptq_out))
    # print("torch vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
    # mean_diff = torch.mean(torch.abs(torch_out - cai_out))
    # max_diff = torch.max(torch.abs(torch_out - cai_out))
    # print("torch vs cai: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))

    # mean_diff = torch.mean(torch.abs(batch_cai_out - batch_gptq_out))
    # max_diff = torch.max(torch.abs(batch_cai_out - batch_gptq_out))
    # print("batch cai vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
    # mean_diff = torch.mean(torch.abs(batch_torch_out - batch_gptq_out))
    # max_diff = torch.max(torch.abs(batch_torch_out - batch_gptq_out))
    # print("batch torch vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
    # mean_diff = torch.mean(torch.abs(batch_torch_out - batch_cai_out))
    # max_diff = torch.max(torch.abs(batch_torch_out - batch_cai_out))
    # print("batch torch vs cai: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))

if __name__ == "__main__":

    test_gptq_linear()

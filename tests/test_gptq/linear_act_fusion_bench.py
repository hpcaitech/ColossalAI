
import torch
import torch.nn as nn

import time
from argparse import ArgumentParser

import transformers
from colossalai.gptq.gptq_utils import GPTQ
from colossalai.gptq.gptq_utils.utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders
from colossalai.gptq.gptq_utils import quant
from colossalai.gptq.gptq_utils.quant import Quantizer
from colossalai.gptq.cai_gptq.gptq_op import CaiGPTQLinearOp
import math
import numpy as np
from colossalai.gptq import CaiInferenceConfig
import csv  

class MLinear(nn.Module):
    def __init__(self, infeature, outfeature):
        super(MLinear, self).__init__()
        self.linear = torch.nn.Linear(infeature, outfeature, dtype=torch.float16)
    def forward(self, x):
        out = self.linear(x)
        return out
    
@torch.no_grad()
def model_quant(model, inps, dev, args):
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

    layers[0] = layers[0].cpu()

    # outs = torch.zeros_like(inps)
    outs = torch.zeros(inps.shape[0], layers[0].linear.weight.shape[0])

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        subset = find_layers(layer)	
        gptq = {}	
        for name in subset:	
            gptq[name] = GPTQ(subset[name])	
            gptq[name].quantizer = Quantizer()	
            gptq[name].quantizer.configure(	args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits	)	
            
        def add_batch(name):	
            def tmp(_, inp, out):	
                gptq[name].add_batch(inp[0].data, out.data)	
            return tmp	
            
        handles = []	
        for name in subset:	
            handles.append(subset[name].register_forward_hook(add_batch(name)))	
           
        for j in range(args.nsamples):	
            outs[j] = layer(inps[j].unsqueeze(0))[0]	
            
        for h in handles:	
            h.remove()	
        for name in subset:	
            print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
            scale,zero,g_idx,error= gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
            quantizers['%s' % (name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
    return quantizers

def model_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        quantizers[name], scale, zero, g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return qlayers['linear']



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
if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--wbits', type=int, default=4, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--nsamples', type=int, default=1, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--groupsize', type=int, default=128, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    args = parser.parse_args()
    infeature = 8192
    outfeature = 8192

    weight = torch.randn(outfeature, infeature).to(torch.float16).to(torch.cuda.current_device())
    bias = torch.zeros(outfeature).to(torch.float16).to(torch.cuda.current_device())
    wn = 16
    ptype = torch.int64
    
    # wn = 8
    # ptype = torch.int32

    qweight = torch.zeros(infeature//wn, outfeature, dtype=ptype, device=torch.cuda.current_device()).contiguous()
    qscales = torch.zeros(infeature//args.groupsize, outfeature, dtype=torch.float16, device=torch.cuda.current_device()).contiguous()
    qzeros = torch.zeros(infeature//args.groupsize, outfeature//wn, dtype=ptype, device=torch.cuda.current_device()).contiguous()


    linear = MLinear(infeature, outfeature)
    linear.to(torch.cuda.current_device())
    with torch.no_grad():
        linear.linear.weight.data.copy_(weight)
        linear.linear.bias.data.copy_(bias)
    inps = torch.randn(1, 1, infeature).to(torch.float16).to(torch.cuda.current_device())
    quantizers = model_quant(linear, inps, torch.cuda.current_device(), args)
    qweight, qscales, qzeros = model_cai_pack(linear, quantizers, qweight, qscales, qzeros, args.wbits, args.groupsize)


    batch_inps = torch.randn(1, 16384, infeature).to(torch.float16).to(torch.cuda.current_device())

    gptq_linear_time = 0
    torch_linear_time = 0
    warm_up_iter = 2
    benchmark_iter = 100

    act_func = nn.ReLU()
    linear.to("cuda")
    for i in range(0, warm_up_iter):
        with torch.no_grad():
            torch_out = act_func(inps)
            # torch_out = inps
            # print(f"torch out {torch_out}")
            torch_out = linear(torch_out)
    torch.cuda.synchronize()

    time_start = time.time()
    for i in range(0, benchmark_iter):
        with torch.no_grad():
            torch_out = act_func(inps)
            # torch_out = inps
            torch_out = linear(torch_out)
    torch.cuda.synchronize()

    time_end = time.time()
    torch_linear_time = time_end - time_start


    time_start = time.time()
    for i in range(0, benchmark_iter):
        with torch.no_grad():
            torch_out = act_func(batch_inps)
            # torch_out = inps
            torch_out = linear(torch_out)
    torch.cuda.synchronize()

    time_end = time.time()
    torch_batch_linear_time = time_end - time_start

    linear.to("cpu")

    gptq_model = model_pack(linear, quantizers, args.wbits, args.groupsize)
    gptq_model.to(torch.cuda.current_device())

    # gptq_model = linear

    for i in range(0, warm_up_iter):
        with torch.no_grad():
            gptq_out = act_func(inps)
            # gptq_out = inps
            gptq_out = gptq_model(gptq_out)
    torch.cuda.synchronize()

    time_start = time.time()
    for i in range(0, benchmark_iter):
        with torch.no_grad():
            gptq_out = act_func(inps)
            # gptq_out = inps
            gptq_out = gptq_model(gptq_out)
    torch.cuda.synchronize()

    time_end = time.time()

    gptq_linear_time = time_end - time_start

    for i in range(0, warm_up_iter):
        with torch.no_grad():
            gptq_out = act_func(batch_inps)
            # gptq_out = inps
            gptq_out = gptq_model(gptq_out)
    torch.cuda.synchronize()

    time_start = time.time()
    for i in range(0, benchmark_iter):
        with torch.no_grad():
            gptq_out = act_func(batch_inps)
            # gptq_out = inps
            gptq_out = gptq_model(gptq_out)
    torch.cuda.synchronize()

    time_end = time.time()

    gptq_batch_linear_time = time_end - time_start

    # qweight = torch.cat((qweight, qweight, qweight), dim=0).contiguous()
    # qscales = torch.cat((qscales, qscales, qscales), dim=0).contiguous()
    # qzeros = torch.cat((qzeros, qzeros, qzeros), dim=0).contiguous()
    # bias = torch.cat((bias, bias, bias), dim=0).contiguous()
    qkv_fused = False
    cai_inf_config = CaiInferenceConfig(fp16=True)

    cai_linear = CaiGPTQLinearOp(cai_inf_config)

    print("cai linear")
    for i in range(0, warm_up_iter):
        with torch.no_grad():
            cai_out = cai_linear(inps,
                                qweight,
                                qscales,
                                qzeros,
                                act_type=0,
                                bias = bias,
                                qkv_fused = qkv_fused)
    torch.cuda.synchronize()


    print("warm up cai linear")

    # f = open('cai_time.csv', 'w')
    # writer = csv.writer(f)


    for i in range(0, warm_up_iter):
        with torch.no_grad():
            cai_out = cai_linear(batch_inps,
                                qweight,
                                qscales,
                                qzeros,
                                act_type=0,
                                bias = bias,
                                qkv_fused = qkv_fused)
    torch.cuda.synchronize()

    cai_linear_time = time_end - time_start
    # print("block dim x:{}, block dim y:{}, time: {:.8f} ".format(i, j, cai_linear_time/benchmark_iter))
    # row=[i, j, cai_linear_time/benchmark_iter]


    time_start = time.time()
    for k in range(0, benchmark_iter):
        with torch.no_grad():
            cai_out = cai_linear(batch_inps,
                                qweight,
                                qscales,
                                qzeros,
                                act_type=0,
                                bias = bias,
                                qkv_fused = qkv_fused)
    torch.cuda.synchronize()
    time_end = time.time()

    batch_cai_linear_time = time_end - time_start

    print("torch time: {:.8f}".format(torch_linear_time/benchmark_iter))
    print("gptq time:{:.8f}".format( gptq_linear_time/benchmark_iter))
    print("cai gptq time:{:.8f}".format( cai_linear_time/benchmark_iter))

    print("batch torch time: {:.8f}".format(torch_batch_linear_time/benchmark_iter))
    print("batch gptq time:{:.8f}".format( gptq_batch_linear_time/benchmark_iter))
    print("batch cai gptq time:{:.8f}".format( batch_cai_linear_time/benchmark_iter))

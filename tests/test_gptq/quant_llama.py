import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from colossalai.gptq.gptq_utils import quant
from colossalai.gptq  import cai_gptq

from colossalai.gptq.gptq_utils import GPTQ, Observer
from colossalai.gptq.gptq_utils.utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
from colossalai.gptq import CaiInferenceConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

import csv

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel
    if args.debug:
        llama_kwargs= {"bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "max_sequence_length": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "pad_token_id": -1,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "use_cache": True,
            "vocab_size": 32000
            }
        configuration = LlamaConfig( **llama_kwargs
        )
        model = LlamaForCausalLM(configuration)
    else:
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)

    # # LlamaForCausalLM
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


# TODO: perform packing on GPU
def cai_llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    # print(f"model {model}")
    # print(f"layers {layers}")

    layers = {n: layers[n] for n in quantizers}
    # print(f"quantizers {quantizers}")
    cai_gptq.make_cai_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [cai_gptq.CaiQuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model

def gptq_llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    # print(f"model {model}")
    # print(f"layers {layers}")

    layers = {n: layers[n] for n in quantizers}
    # print(f"quantizers {quantizers}")
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def cai_load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    cai_gptq.make_cai_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    print('Done.')

    return model


def gptq_load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    print('Done.')

    return model

all_perfs = []
now_perf=[]

def print_perf_stats(latency_set, config, warmup=3):
    global now_perf
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2
        # if args.dtype == "float16":
        #     num_bytes = 2
        # elif args.dtype == "float32":
        #     num_bytes = 4
        # else:
        #     num_bytes = 1
        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * args.batch_size / 1e12))
        print("Alloc GPU Mem: {0:8.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
        print("Max alloc GPU Mem: {0:8.2f} GB".format(torch.cuda.max_memory_allocated()/1e9))
        row = [args.batch_size, args.input_len, args.max_new_tokens, "{0:8.2f}".format(avg * 1000), 
                "{0:8.2f}".format(torch.cuda.memory_allocated() / 1e9),
                "{0:8.2f}".format(torch.cuda.max_memory_allocated()/1e9)]
        with open('./{}_profile.csv'.format(args.model_type), 'a', encoding='UTF8') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            writer.writerow(row)

        now_perf.append("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        now_perf.append("Alloc GPU Mem: {0:8.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
        now_perf.append("Max alloc GPU Mem: {0:8.2f} GB".format(torch.cuda.max_memory_allocated()/1e9))

        all_perfs.append(now_perf)
        now_perf = []

def benchmark(model):

    input_tokens = {"input_ids":torch.randint(1, 1000, (args.batch_size, args.input_len), device=DEV), 
        "attention_mask":torch.ones((args.batch_size, args.input_len), device=DEV)}
    torch.cuda.synchronize()
    iters = 10 if args.benchmark else 2 #warmup
    print(f"model config {model.config}")

    times = []
    warmup=3
    prof_flag = 0
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.reset_peak_memory_stats()
    for i in range(iters):
        if i >= warmup:
            prof_flag=1
        torch.cuda.synchronize()
        start = time.time()
        outputs = model.generate(**input_tokens,
                **generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    print("outpus shape: ", outputs.shape)
    print(args)
    print("input batch, input len, out len: ",args.batch_size, args.input_len, args.max_new_tokens)
    # if args.local_rank == 0:
    now_perf.extend(["input batch, input len, out len: ",args.batch_size, args.input_len, args.max_new_tokens])
    print_perf_stats(map(lambda t: t / args.max_new_tokens, times), model.config)

def test(model_1, model_2):
    # input_tokens = {"input_ids":torch.randint(1, 1000, (args.batch_size, args.input_len), device=DEV), 
    #     "attention_mask":torch.ones((args.batch_size, args.input_len), device=DEV)}
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)


    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.unk_token_id

    text = "how is weather today? I want to know the weather of beijing. "
    text = "how are you?"

    inputs = [text]
    input_tokens = tokenizer.batch_encode_plus(inputs, padding = True, return_tensors="pt")

    input_len = 0
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
            # print(input_tokens[t].shape)
            input_len = input_tokens[t].shape[1]

    outputs_1 = model_1.generate(**input_tokens,
            **generate_kwargs)
    print("model 1 done")
    out_1 =  tokenizer.batch_decode(outputs_1)

    print("decode out:", out_1)
    if model_2 is None:
        return
    outputs_2 = model_2.generate(**input_tokens,
            **generate_kwargs)
    print("model 2 done")

    out_2 =  tokenizer.batch_decode(outputs_2)

    ret = torch.allclose(outputs_1, outputs_2)
    print("allclose is ", ret)

    print("decode out:", out_2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=1, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', action='store_true', help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='Max new tokens to generate.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to generate.')
    parser.add_argument('--input_len', type=int, default=128, help='Batch size to generate.')
    parser.add_argument('--model_type', type=str, choices=['cai', 'gptq', 'torch'], default='torch', help='Batch size to generate.')
    parser.add_argument('--debug', action='store_true', help='Whether to debug or not')

    args = parser.parse_args()

    model_packed = False
    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        if args.model_type == "gptq":
            model = gptq_load_quant(args.model, args.load, args.wbits, args.groupsize)
        elif args.model_type == "cai":
            model = cai_load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_llama(args.model)
    model.half()

    if not args.load and args.wbits < 16 and not args.nearest and args.model_type in ['cai', 'gptq']:
        dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        if args.model_type == "cai":
            cai_llama_pack(model, quantizers, args.wbits, args.groupsize)
        elif args.model_type == "gptq":
            gptq_llama_pack(model, quantizers, args.wbits, args.groupsize)
        model_packed = True      
        print(time.time() - tick)


    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save and args.model_type in ['cai', 'gptq']:
        if not model_packed:
            llama_pack(model, quantizers, args.wbits, args.groupsize)
            model_packed = True
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors and args.model_type in ['cai', 'gptq']:
        if not model_packed:
            llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

    if args.benchmark:
        # model = model.to(DEV)
        # print(f"model config {model.config.num_key_value_heads}")

        # if args.model_type == "cai":
        #     cai_inf_config = CaiInferenceConfig(fp16=True, 
        #                                         device=torch.cuda.current_device(), 
        #                                         gptq=True,
        #                                         gptq_group_size=128,
        #                                         gptq_quant_bits=4)
        #     model = convert_to_ds_model(model, cai_inf_config)
        # model.cuda().to(torch.cuda.current_device())


        torch_model = get_llama(args.model)
        torch_model.half()
        torch_model = torch_model.to(DEV)

        gptq_model = gptq_load_quant(args.model, "llama7b-4bit-128g-gptq-nao.pt", args.wbits, args.groupsize)
        gptq_model = gptq_model.to(DEV)

        model = cai_load_quant(args.model, args.load, args.wbits, args.groupsize)
        model = model.to(DEV)


        test(torch_model, model)
        test(gptq_model, None)

        print("torch_model ", torch_model)
        print("gptq_model ", gptq_model)
        print("cai_model ", model)
        torch_qkv_out = torch_model.model.layers[0].self_attn.qkv_out
        cai_qkv_out = model.model.layers[0].self_attn.qkv_out
        gptq_qkv_out = gptq_model.model.layers[0].self_attn.qkv_out

        gptq_out = gptq_model.model.layers[0].self_attn.q_proj.scales
        cai_out = model.model.layers[0].self_attn.q_proj.scales

        mean_diff = torch.mean(torch.abs(cai_out - gptq_out))
        max_diff = torch.max(torch.abs(cai_out - gptq_out))
        print("cai vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
        for i in range(3):
            cai_out = cai_qkv_out[i]
            torch_out = torch_qkv_out[i]
            gptq_out = gptq_qkv_out[i]
            mean_diff = torch.mean(torch.abs(cai_out - gptq_out))
            max_diff = torch.max(torch.abs(cai_out - gptq_out))
            print("cai vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
            mean_diff = torch.mean(torch.abs(torch_out - gptq_out))
            max_diff = torch.max(torch.abs(torch_out - gptq_out))
            print("torch vs gptq: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))
            mean_diff = torch.mean(torch.abs(torch_out - cai_out))
            max_diff = torch.max(torch.abs(torch_out - cai_out))
            print("torch vs cai: mean_diff=%.8f, max_diff=%.8f" % (mean_diff, max_diff))

        # # for batch in [1, 2, 4, 8, 16, 32]:
        # for batch in [1]:
        #     args.batch_size = batch
        #     # for in_len in [128, 256, 512, 1024, 2048]:
        #     for in_len in [1024]:
        #         args.input_len = in_len
        #         benchmark(model)
        #     # for info in all_perfs:
        #     #     print(info)
        #     # # all_perfs = []
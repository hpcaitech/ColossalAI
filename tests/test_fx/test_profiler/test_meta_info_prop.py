from time import sleep, time
import copy
from typing import Tuple, List, Union, Optional
from colossalai.fx.profiler.memory import activation_size, calculate_fwd_tmp, is_inplace, calculate_fwd_out
from colossalai.fx.tracer.tracer import ColoTracer
import torch
import torch.nn as nn
import torchvision.models as tm
from torch.fx import symbolic_trace
import torch.fx
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import MetaTensor, parameter_size

from gpt_utils import GPTLMModel, gpt2_medium, get_data, GPTLMLoss, gpt2_xl

BATCH_SIZE = 16
NUM_STEPS = 1


def extract_forward_mem(gm: torch.fx.GraphModule):
    node_size = 0
    param_size = 0
    for node in gm.graph.nodes:
        node_size += calculate_fwd_tmp(node)
        node_size += calculate_fwd_out(node)
    param_size = parameter_size(gm)
    return (node_size + param_size) / 1024**2, param_size / 1024**2


def extract_forward_flops(gm: torch.fx.GraphModule):
    fwd_flop = 0
    bwd_flop = 0
    for node in gm.graph.nodes:
        fwd_flop += node.meta.get('fwd_flop', 0)
        bwd_flop += node.meta.get('bwd_flop', 0)
    return fwd_flop, bwd_flop


def gen_tm_data(batch_size: int, shape: Tuple[int, int, int], device='cuda'):
    data = torch.rand(batch_size, *shape, device=device)
    label = torch.empty(batch_size, dtype=torch.long, device=device).random_(1000)
    return data, label


def test_tm_forward(gm: torch.fx.GraphModule):
    torch.cuda.reset_peak_memory_stats()
    forward_mem = -torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    param_mem = -torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    gm.cuda()
    param_mem += torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    optimizer = Adam(gm.parameters(), lr=1e-3)
    gm.train()
    for n in range(NUM_STEPS):
        data, _ = gen_tm_data(BATCH_SIZE, (3, 224, 224))

        # If we need to dive deep into the memory usage by
        # inspecting `saved_tensor_hooks`

        # =====================================================
        # fwd_mem = 0
        # cache = set()
        # def pack(x):
        #     if isinstance(x, torch.Tensor):
        #         nonlocal fwd_mem, cache
        #         if x.data_ptr() not in cache:
        #             fwd_mem += activation_size(x)
        #             cache.add(x.data_ptr())
        #     return x
        # def unpack(x):
        #     return x
        #
        # with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        #    output = gm(data)
        # =====================================================
        output = gm(data)
        optimizer.zero_grad()
        forward_mem += torch.cuda.memory_allocated(device="cuda:0") / 1024**2 / NUM_STEPS
    return forward_mem, param_mem


def test_gpt_forward(gm: torch.fx.GraphModule, num_steps: int = 5):

    def get_gpu_mem():
        result = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        return result

    # get_gpu_mem()   # reset
    forward_mem = 0
    param_mem = 0
    gm = gpt2_medium()
    gm.train()
    gm.cuda()
    param_mem += get_gpu_mem()
    time_0 = time()
    # criterion = GPTLMLoss()
    # optimizer = Adam(gm.parameters(), lr=1e-3)
    for n in range(num_steps):
        data, mask = get_data(8, 1024, 50257, device='cuda')
        fwd_mem = 0
        unpack_mem = 0
        i = 0
        cache = set()

        def pack(x):
            if isinstance(x, torch.Tensor):
                nonlocal i
                nonlocal fwd_mem
                fwd_mem += activation_size(x)
            # if isinstance(x, torch.Tensor):
            # print(type(x), activation_size(x), x.shape, x.dtype)
            return x

        def unpack(x):
            nonlocal unpack_mem
            unpack_mem += activation_size(x)
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            output = gm(data, mask)
        forward_mem += get_gpu_mem() / num_steps
        # optimizer.zero_grad()
        # loss = criterion(output, data)
        # loss.backward()
        # optimizer.step()
        get_gpu_mem()    # reset
        print((fwd_mem) // 1024**2 + param_mem, unpack_mem / 1024**2)
    print(time() - time_0)
    return forward_mem, param_mem


def test_meta_info_prop():
    for m in [
            tm.alexnet, tm.resnet18, tm.resnet34, tm.resnet50, tm.resnet101, tm.resnet152, tm.densenet121,
            tm.densenet161, tm.densenet169, tm.densenet201, tm.convnext_tiny, tm.convnext_small, tm.convnext_base,
            tm.convnext_large, tm.wide_resnet50_2, tm.wide_resnet101_2
    ]:
        # for M in [tm.regnet_x_16gf, tm.mnasnet0_5, tm.efficientnet_b0]:
        # for M in [tm.convnext_tiny]:
        # for M in [tm.shufflenet_v2_x0_5, tm.shufflenet_v2_x1_0, tm.shufflenet_v2_x1_5, tm.shufflenet_v2_x2_0]:
        # for M in [tm.shufflenet_v2_x0_5]:
        # for M in [MyNet]:
        # for M in [tm.mobilenet_v2, tm.mobilenet_v3_small, tm.mobilenet_v3_large]:
        # for M in [tm.mobilenet_v3_small]:
        # for M in [gpt2_medium]:
        # for M in [tm.resnext50_32x4d, tm.resnext101_32x8d, tm.resnext101_64x4d]:
        # for M in [tm.vit_b_16, tm.vit_b_32, tm.vit_h_14, tm.vit_l_16, tm.vit_l_32]:
        # for M in [tm.vit_b_16]:
        # for M in [tm.vgg11]:
        model = m()
        model.train()
        data = MetaTensor(torch.rand(int(BATCH_SIZE), 3, 224, 224, device='meta'), fake_device='cuda')
        gm = symbolic_trace(model)
        gm: torch.fx.GraphModule
        # gm.graph.print_tabular()
        # graph = meta_trace(gm, data)
        # print(graph.python_code('self').src)
        interp = MetaInfoProp(gm.cuda())
        time_0 = time()
        interp.run(data)
        # print(interp.summary())
        print(time() - time_0)
        gm.cpu()

        meta_forward_mem, meta_param_mem = _forward_mem(gm)
        fwd_flop, bwd_flop = _forward_flops(gm)
        time_0 = time()
        concrete_forward_mem, concrete_param_mem = test_forward(gm, num_steps=1)
        print(time() - time_0)

        print(
            f'|{M}|{meta_forward_mem:.3f} MB|{meta_param_mem:.3f} MB|{concrete_forward_mem:.3f} MB|{concrete_param_mem:.3f} MB|fwd_flop={fwd_flop / 1e9:.3f}GFLOPs|bwd_flop={bwd_flop / 1e9:.3f}GFLOPs|'
        )
        # sleep(2)
        del model, gm


def test_gpt_meta_info_prop():
    # for M in [tm.resnet18, tm.resnet34, tm.resnet50, tm.resnet101, tm.resnet152]:
    # for M in [tm.resnet18]:
    # for M in [tm.densenet121, tm.densenet161, tm.densenet169, tm.densenet201]:
    # for M in [tm.convnext_tiny, tm.convnext_small, tm.convnext_base, tm.convnext_large]:
    # for M in [tm.wide_resnet50_2, tm.wide_resnet101_2]:
    # for M in [tm.regnet_x_16gf, tm.mnasnet0_5, tm.efficientnet_b0]:
    # for M in [tm.convnext_tiny]:
    # for M in [tm.shufflenet_v2_x0_5, tm.shufflenet_v2_x1_0, tm.shufflenet_v2_x1_5, tm.shufflenet_v2_x2_0]:
    # for M in [tm.shufflenet_v2_x0_5]:
    # for M in [MyNet]:
    # for M in [tm.mobilenet_v2, tm.mobilenet_v3_small, tm.mobilenet_v3_large]:
    for M in [gpt2_medium]:
        # for M in [tm.resnext50_32x4d, tm.resnext101_32x8d, tm.resnext101_64x4d]:
        # for M in [tm.vit_b_16, tm.vit_b_32, tm.vit_h_14, tm.vit_l_16, tm.vit_l_32]:
        # for M in [tm.vit_b_16]:
        model = M().cuda()
        model.train()
        data, mask = get_data(8, 1024, 50257, device='meta')
        print(activation_size((data, mask)) / 1024**2)
        graph = ColoTracer().trace(model, meta_args={'input_ids': data, 'attention_mask': mask})
        gm = torch.fx.GraphModule(model, graph)
        # gm = symbolic_trace(model)
        gm: torch.fx.GraphModule
        # gm.graph.print_tabular()
        # gm_meta = copy.deepcopy(gm).to('meta')
        # graph = meta_trace(gm, data, mask)
        # print(graph.python_code('self').src)
        interp = MetaInfoProp(gm.cuda())
        time_0 = time()
        interp.run(MetaTensor(data, fake_device='cuda:0'), MetaTensor(mask, fake_device='cuda:0'))
        print(time() - time_0)
        # print(interp.summary())
        model.cpu()

        fwd_flop, bwd_flop = _forward_flops(gm)

        concrete_forward_mem, concrete_param_mem = test_gpt_forward(gm, num_steps=1)
        meta_forward_mem, meta_param_mem = _forward_mem(gm)

        print(
            f'|{M}|{meta_forward_mem:.3f} MB|{meta_param_mem:.3f} MB|{concrete_forward_mem:.3f} MB|{concrete_param_mem:.3f} MB|fwd_flop={fwd_flop / 1e9:.3f}GFLOPs|bwd_flop={bwd_flop / 1e9:.3f}GFLOPs|'
        )
        # sleep(2)
        del model


if __name__ == '__main__':
    # test_meta_info_prop()
    test_gpt_meta_info_prop()

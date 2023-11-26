from typing import Tuple

import torch
import torch.fx
import torchvision.models as tm
from gpt_utils import gpt2_medium
from torch.fx import symbolic_trace

from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp, is_compatible_with_meta, parameter_size
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.testing import clear_cache_before_run, run_on_environment_flag

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

TM_BATCH_SIZE = 64
GPT_BATCH_SIZE = 8
NUM_STEPS = 5


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
        fwd_flop += node.meta.get("fwd_flop", 0)
        bwd_flop += node.meta.get("bwd_flop", 0)
    return fwd_flop, bwd_flop


def gen_tm_data(batch_size: int, shape: Tuple[int, int, int], device="cuda"):
    data = torch.rand(batch_size, *shape, device=device)
    label = torch.empty(batch_size, dtype=torch.long, device=device).random_(1000)
    return data, label


def gen_gpt_data(batch_size, seq_len, vocab_size, device="cpu"):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return input_ids, attention_mask


def run_tm_forward(gm: torch.fx.GraphModule):
    torch.cuda.reset_peak_memory_stats()
    forward_mem = -torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    param_mem = -torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    gm.cuda()
    param_mem += torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    gm.train()
    for n in range(NUM_STEPS):
        torch.cuda.reset_peak_memory_stats()
        data, _ = gen_tm_data(TM_BATCH_SIZE, (3, 224, 224))

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
        # print(f'Memory estimation by saved_tensor_hooks: {fwd_mem / 1024**2}')
        # =====================================================

        output = gm(data)
        forward_mem += torch.cuda.memory_allocated(device="cuda:0") / 1024**2 / NUM_STEPS
        del output
    return forward_mem, param_mem


def run_gpt_forward(gm: torch.fx.GraphModule):
    torch.cuda.reset_peak_memory_stats()
    forward_mem = -torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    param_mem = -torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    gm.cuda()
    param_mem += torch.cuda.memory_allocated(device="cuda:0") / 1024**2
    for n in range(NUM_STEPS):
        torch.cuda.reset_peak_memory_stats()
        data, mask = gen_gpt_data(GPT_BATCH_SIZE, 1024, 50257, device="cuda:0")

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
        #    output = gm(data, mask)
        # print(f'Memory estimation by saved_tensor_hooks: {fwd_mem / 1024**2}')
        # =====================================================

        output = gm(data, mask)
        forward_mem += torch.cuda.memory_allocated(device="cuda:0") / 1024**2 / NUM_STEPS
        del output
    return forward_mem, param_mem


@run_on_environment_flag(name="FX_PROFILER")
@clear_cache_before_run()
def test_meta_info_prop():
    for m in [
        tm.alexnet,
        tm.resnet18,
        tm.resnet34,
        tm.resnet50,
        tm.resnet101,
        tm.resnet152,
        tm.densenet121,
        tm.densenet161,
        tm.densenet169,
        tm.densenet201,
        tm.convnext_tiny,
        tm.convnext_small,
        tm.convnext_base,
        tm.convnext_large,
        tm.wide_resnet50_2,
        tm.wide_resnet101_2,
        tm.regnet_x_16gf,
        tm.mnasnet0_5,
        tm.efficientnet_b0,
        tm.shufflenet_v2_x0_5,
        tm.shufflenet_v2_x1_0,
        tm.shufflenet_v2_x1_5,
        tm.shufflenet_v2_x2_0,
        tm.mobilenet_v2,
        tm.mobilenet_v3_small,
        tm.mobilenet_v3_large,
        tm.resnext50_32x4d,
        tm.resnext101_32x8d,
        tm.resnext101_64x4d,
        tm.vit_b_16,
        tm.vit_b_32,
        tm.vit_h_14,
        tm.vit_l_16,
        tm.vit_l_32,
        tm.vgg11,
        tm.vgg11_bn,
        tm.vgg13,
        tm.vgg13_bn,
        tm.vgg16,
        tm.vgg16_bn,
        tm.vgg19,
        tm.vgg19_bn,
    ]:
        model = m().cuda()
        model.train()
        data = MetaTensor(torch.rand(int(TM_BATCH_SIZE), 3, 224, 224, device="meta"), fake_device="cuda:0")
        gm = symbolic_trace(model)
        interp = MetaInfoProp(gm)
        interp.propagate(data)
        gm.cpu()

        meta_forward_mem, meta_param_mem = extract_forward_mem(gm)
        fwd_flop, bwd_flop = extract_forward_flops(gm)
        concrete_forward_mem, concrete_param_mem = run_tm_forward(gm)

        print(
            f"|{m.__name__}|{meta_forward_mem:.3f} MB|{meta_param_mem:.3f} MB|{concrete_forward_mem:.3f} MB|{concrete_param_mem:.3f} MB|fwd_flop={fwd_flop / 1e9:.3f}GFLOPs|bwd_flop={bwd_flop / 1e9:.3f}GFLOPs|"
        )
        del model, gm


@run_on_environment_flag(name="FX_PROFILER")
@clear_cache_before_run()
def test_gpt_meta_info_prop():
    for m in [gpt2_medium]:
        model = m().cuda()
        model.train()
        data, mask = gen_gpt_data(GPT_BATCH_SIZE, 1024, 50257, device="meta")
        graph = ColoTracer().trace(model, meta_args={"input_ids": data, "attention_mask": mask})
        gm = torch.fx.GraphModule(model, graph)
        interp = MetaInfoProp(gm)
        interp.propagate(MetaTensor(data, fake_device="cuda:0"), MetaTensor(mask, fake_device="cuda:0"))
        model.cpu()

        fwd_flop, bwd_flop = extract_forward_flops(gm)

        concrete_forward_mem, concrete_param_mem = run_gpt_forward(gm)
        meta_forward_mem, meta_param_mem = extract_forward_mem(gm)

        print(
            f"|{m.__name__}|{meta_forward_mem:.3f} MB|{meta_param_mem:.3f} MB|{concrete_forward_mem:.3f} MB|{concrete_param_mem:.3f} MB|fwd_flop={fwd_flop / 1e9:.3f}GFLOPs|bwd_flop={bwd_flop / 1e9:.3f}GFLOPs|"
        )
        del model, gm


if __name__ == "__main__":
    test_meta_info_prop()
    test_gpt_meta_info_prop()

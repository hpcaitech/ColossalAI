import time
from typing import Any, Dict, List

import torch
import torch.fx

import colossalai
from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.utils import free_port

if AUTOCHUNK_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def _benchmark_autochunk_gpt_gm(
    model: Any,
    data: tuple,
    max_memory: int = None,
) -> None:
    model = model.cuda().eval()

    # build model and input
    meta_args, concrete_args, sequence = data
    if concrete_args is None:
        concrete_args = {}

    # trace the meta graph and setup codegen
    meta_graph = symbolic_trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args.items()},
        concrete_args={k: v for k, v in concrete_args.items()},
    )
    interp = MetaInfoProp(meta_graph)
    meta_tensors = [meta_args[i] if i in meta_args else concrete_args[i] for i in sequence]
    meta_tensors = [MetaTensor(i, fake_device="cuda:0") if isinstance(i, torch.Tensor) else i for i in meta_tensors]
    interp.propagate(*meta_tensors)
    codegen = AutoChunkCodeGen(
        meta_graph,
        max_memory=max_memory,
    )

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(
        model.cuda().eval(),
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args.items()},
        concrete_args={k: v for k, v in concrete_args.items()},
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # init inputs
    inputs = [meta_args[i] if i in meta_args else concrete_args[i] for i in sequence]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda().eval()

    # bench
    mem = _benchmark_memory(gm, inputs)
    speed = _benchmark_speed(gm, inputs)
    print("gpt gm, mem: %.2fMB, time: %.4fs" % (mem, speed))


def _benchmark_autochunk_gpt_origin(
    model: Any,
    data: tuple,
) -> None:
    # build model and input
    meta_args, concrete_args, sequence = data
    if concrete_args is None:
        concrete_args = {}

    # init inputs
    inputs = [meta_args[i] if i in meta_args else concrete_args[i] for i in sequence]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda().eval()

    # bench
    mem = _benchmark_memory(model, inputs)
    speed = _benchmark_speed(model, inputs)
    print("gpt origin, mem: %.2fMB, time: %.4fs" % (mem, speed))


def _benchmark_memory(model, inputs):
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        now_mem = float(torch.cuda.memory_allocated()) / 1024**2
        model(*inputs)
        new_max_mem = float(torch.cuda.max_memory_allocated()) / 1024**2
    return new_max_mem - now_mem


def _benchmark_speed(model, inputs, loop=5):
    with torch.no_grad():
        for _ in range(loop // 2 + 1):
            model(*inputs)
        torch.cuda.synchronize()
        time1 = time.time()
        for _ in range(loop):
            model(*inputs)
        torch.cuda.synchronize()
        time2 = time.time()
    return (time2 - time1) / loop


def benchmark_autochunk_gpt():
    from test_autochunk_gpt import GPT2Config, GPT2Model, get_data

    batch = 1
    seq = 512
    n_embd = 96

    model = GPT2Model
    config = GPT2Config(n_embd=n_embd, n_position=seq, n_layer=2, n_head=4)
    model = model(config=config)
    shape = [batch, seq]
    print("")
    _benchmark_autochunk_gpt_origin(model, get_data(shape))
    _benchmark_autochunk_gpt_gm(model, get_data(shape), None)


if __name__ == "__main__":
    # launch colossalai
    colossalai.launch(
        config={},
        rank=0,
        world_size=1,
        host="localhost",
        port=free_port(),
        backend="nccl",
    )
    benchmark_autochunk_gpt()

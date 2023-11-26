import time
from typing import Any

import torch
import torch.fx

import colossalai
from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.testing import free_port

if AUTOCHUNK_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def _benchmark_evoformer_stack_gm(
    data_args: tuple,
    max_memory: int,
    get_model: Any,
    get_data: Any,
) -> None:
    # build model and input
    model = get_model().cpu().eval()
    meta_args, concrete_args = get_data(*data_args)
    if concrete_args is None:
        concrete_args = []

    # trace the meta graph and setup codegen
    meta_graph = symbolic_trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    interp = MetaInfoProp(meta_graph)
    meta_tensors = [MetaTensor(i[1], fake_device="cpu") for i in meta_args] + [i[1] for i in concrete_args]
    interp.propagate(*meta_tensors)
    codegen = AutoChunkCodeGen(
        meta_graph,
        max_memory=max_memory,
    )

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # init inputs
    inputs = [i[1] for i in meta_args] + [i[1] for i in concrete_args]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda()

    # bench
    mem = _benchmark_memory(gm, inputs)
    speed = _benchmark_speed(gm, inputs)
    print("evoformer stack gm, mem: %.2fMB, time: %.4fs" % (mem, speed))


def _benchmark_evoformer_stack_origin(
    data_args: tuple,
    get_model: Any,
    get_data: Any,
) -> None:
    # build model and input
    model = get_model()
    meta_args, concrete_args = get_data(*data_args)
    if concrete_args is None:
        concrete_args = []

    # init inputs
    inputs = [i[1] for i in meta_args] + [i[1] for i in concrete_args]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda()

    # bench
    mem = _benchmark_memory(model, inputs)
    speed = _benchmark_speed(model, inputs)
    print("evoformer stack origin, mem: %.2fMB, time: %.4fs" % (mem, speed))
    return mem


def _benchmark_memory(model, inputs):
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        now_mem = torch.cuda.memory_allocated() / 1024**2
        model(*inputs)
        new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
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


def benchmark_evoformer_stack(data_args):
    from test_autochunk_evoformer_stack import get_data, get_model

    print("\nmsa len: %d, pair len: %d" % (data_args[0], data_args[1]))
    max_mem = _benchmark_evoformer_stack_origin(data_args, get_model, get_data)
    for ratio in [0.5, 0.4, 0.3, 0.2, 0.1]:
        try:
            _benchmark_evoformer_stack_gm(data_args, max_mem * ratio, get_model, get_data)
        except RuntimeError as e:
            if e.args[0] == "Search failed. Try a larger memory threshold.":
                break
        except Exception as e:
            raise e
    _benchmark_evoformer_stack_gm(data_args, None, get_model, get_data)


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
    benchmark_evoformer_stack((256, 256))
    benchmark_evoformer_stack((256, 512))
    benchmark_evoformer_stack((256, 1024))
    benchmark_evoformer_stack((256, 1280))

from typing import Any, Dict, List

import torch
import torch.fx

import colossalai
from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.legacy.core import global_context as gpc

if AUTOCHUNK_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def assert_codegen_run(
    model: Any,
    meta_args: List,
    concrete_args: List = None,
    max_memory: int = None,
    print_mem: bool = False,
    print_est_mem: bool = False,
    print_progress: bool = False,
    print_code: bool = False,
) -> List[Dict]:
    if concrete_args is None:
        concrete_args = []
    model = model()

    # trace the meta graph and setup codegen
    meta_graph = symbolic_trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    model = model.cuda().eval()
    interp = MetaInfoProp(meta_graph)
    meta_tensors = [MetaTensor(i[1], fake_device="cuda:0") for i in meta_args] + [i[1] for i in concrete_args]
    interp.propagate(*meta_tensors)
    codegen = AutoChunkCodeGen(
        meta_graph,
        max_memory=max_memory,
        print_mem=print_est_mem,
        print_progress=print_progress,
    )
    chunks = codegen.chunk_infos

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(
        model.cuda(),
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # assert chunk in code
    code = graph.python_code("self").src
    if print_code:
        print(code)
    assert "chunk_size = None;  " in code

    # assert result
    inputs = [i[1] for i in meta_args] + [i[1] for i in concrete_args]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda().eval()
    gm.eval()
    with torch.no_grad():
        if print_mem:
            torch.cuda.reset_peak_memory_stats()
            now_mem_gm = torch.cuda.memory_allocated() / 1024**2
        out_gm = gm(*[i.clone() if isinstance(i, torch.Tensor) else i for i in inputs])
        if print_mem:
            max_mem_gm = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()
            now_mem_ori = torch.cuda.memory_allocated() / 1024**2
        out_model = model(*[i.clone() if isinstance(i, torch.Tensor) else i for i in inputs])
        if print_mem:
            max_mem_ori = torch.cuda.max_memory_allocated() / 1024**2
            print("origin mem: %.2fMB, autochunk mem: %.2fMB" % (max_mem_ori - now_mem_ori, max_mem_gm - now_mem_gm))

    assert torch.allclose(
        out_gm["sample"], out_model["sample"], atol=1e-3
    ), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
        torch.abs(out_gm["sample"] - out_model["sample"])
    )

    return chunks


def run_test(
    rank: int,
    world_size: int,
    port: int,
    model: Any,
    data: tuple,
    max_memory: int,
    print_code: bool = False,
    print_mem: bool = False,
    print_est_mem: bool = False,
    print_progress: bool = False,
    get_chunk_target: Any = None,
) -> None:
    # launch colossalai
    colossalai.launch(
        config={},
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )

    # build model and input
    meta_args, concrete_args = data
    chunks = assert_codegen_run(
        model,
        meta_args=meta_args,
        concrete_args=concrete_args,
        max_memory=max_memory,
        print_code=print_code,
        print_mem=print_mem,
        print_est_mem=print_est_mem,
        print_progress=print_progress,
    )

    if get_chunk_target is not None:
        chunk_found = [i["region"] for i in chunks]
        chunk_target = get_chunk_target()[max_memory]
        assert chunk_found == chunk_target, "found regions %s doesn't equal target regions %s" % (
            str(chunk_found),
            str(chunk_target),
        )

    gpc.destroy()

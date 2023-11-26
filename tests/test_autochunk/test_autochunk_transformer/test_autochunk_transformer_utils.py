from typing import Any, Dict, List

import torch
import torch.fx

import colossalai
from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp

if AUTOCHUNK_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def assert_codegen_run(
    model: Any,
    data: tuple,
    max_memory: int = None,
    print_est_mem: bool = False,
    print_mem: bool = False,
    print_progress: bool = False,
    print_code: bool = False,
    eval_mem: bool = False,
) -> List[Dict]:
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
        meta_graph, max_memory=max_memory, print_mem=print_est_mem, print_progress=print_progress, eval_mem=eval_mem
    )
    chunks = codegen.chunk_infos

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(
        model.cuda(),
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args.items()},
        concrete_args={k: v for k, v in concrete_args.items()},
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
    inputs = [meta_args[i] if i in meta_args else concrete_args[i] for i in sequence]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda().eval()
    gm.eval()
    with torch.no_grad():
        if print_mem:
            torch.cuda.reset_peak_memory_stats()
            now_mem = torch.cuda.memory_allocated() / 1024**2
        out_gm = gm(*[i.clone() if isinstance(i, torch.Tensor) else i for i in inputs])
        if print_mem:
            new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
            print("mem: %.2fMB" % (new_max_mem - now_mem))
        out_model = model(*inputs)
    assert_allclose(out_model, out_gm)
    return chunks


def assert_allclose(out_model: Any, out_gm: Any) -> None:
    """
    assert allclose for out
    """
    if isinstance(out_model, torch.Tensor):
        assert torch.allclose(
            out_model, out_gm, atol=1e-4
        ), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(torch.abs(out_model - out_gm))
    elif isinstance(out_model, dict):
        for k in out_model.keys():
            assert_allclose(out_model[k], out_gm[k])
    elif isinstance(out_model, tuple) or isinstance(out_model, list) or isinstance(out_model, set):
        for i, j in zip(out_model, out_gm):
            assert_allclose(i, j)


def run_test(
    rank: int,
    world_size: int,
    port: int,
    model: Any,
    config: Any,
    data: tuple,
    max_memory: int,
    print_code: bool = False,
    print_est_mem: bool = False,
    print_mem: bool = False,
    print_progress: bool = False,
    eval_mem: bool = False,
    get_chunk_target: Any = None,
) -> None:
    model = model(config=config)
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
    chunks = assert_codegen_run(
        model,
        data=data,
        max_memory=max_memory,
        print_code=print_code,
        print_est_mem=print_est_mem,
        print_mem=print_mem,
        print_progress=print_progress,
        eval_mem=eval_mem,
    )

    if get_chunk_target is not None:
        chunk_found = [i["region"] for i in chunks]
        chunk_target = get_chunk_target()[max_memory]
        assert chunk_found == chunk_target, "found regions %s doesn't equal target regions %s" % (
            str(chunk_found),
            str(chunk_target),
        )

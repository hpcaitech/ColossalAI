from typing import Any, Dict, List

import torch
import torch.fx

import colossalai
from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.core import global_context as gpc
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.utils import free_port

if AUTOCHUNK_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def assert_codegen_run(
    model: Any,
    data: tuple,
    max_memory: int = None,
    print_mem: bool = False,
    print_progress: bool = False,
    print_code: bool = False,
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
        meta_graph,
        max_memory=max_memory,
        print_mem=print_mem,
        print_progress=print_progress,
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
    assert "chunk_result = None;  chunk_size = None;" in code

    # assert result
    inputs = [meta_args[i] if i in meta_args else concrete_args[i] for i in sequence]
    inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]
    model.cuda().eval()
    gm.eval()
    with torch.no_grad():
        out_gm = gm(*inputs)
        out_model = model(*inputs)
    for k in out_model.keys():
        if torch.is_tensor(out_gm[k]):
            assert torch.equal(
                out_model[k], out_gm[k]
            ), f'{model.__class__.__name__} has incorrect output {k}, expect {out_model[k]}, but got {out_gm[k]}'

    return chunks


def run_test(
    rank: int,
    model: Any,
    config: Any,
    data: tuple,
    max_memory: int,
    print_code: bool,
    print_mem: bool,
    print_progress: bool,
    get_chunk_target: Any = None,
) -> None:
    model = model(config=config)
    # launch colossalai
    colossalai.launch(
        config={},
        rank=rank,
        world_size=1,
        host="localhost",
        port=free_port(),
        backend="nccl",
    )

    # build model and input
    chunks = assert_codegen_run(
        model,
        data=data,
        max_memory=max_memory,
        print_code=print_code,
        print_mem=print_mem,
        print_progress=print_progress,
    )

    if get_chunk_target is not None:
        chunk_found = [i["region"] for i in chunks]
        chunk_target = get_chunk_target()[max_memory]
        assert (chunk_found == chunk_target), "found regions %s doesn't equal target regions %s" % (
            str(chunk_found),
            str(chunk_target),
        )

    gpc.destroy()

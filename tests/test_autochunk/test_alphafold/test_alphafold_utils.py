from typing import Any, Dict, List

import torch
import torch.fx

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp

if AUTOCHUNK_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def assert_codegen_run(model: Any,
                       meta_args: List,
                       concrete_args: List = None,
                       max_memory: int = None,
                       print_mem: bool = False,
                       print_progress: bool = False,
                       print_code: bool = False) -> List[Dict]:
    if concrete_args is None:
        concrete_args = []

    # trace the meta graph and setup codegen
    meta_graph = symbolic_trace(
        model,
        meta_args={k: v.to(torch.device("meta")) for k, v in meta_args},
        concrete_args={k: v for k, v in concrete_args},
    )
    interp = MetaInfoProp(meta_graph)
    meta_tensors = [MetaTensor(i[1], fake_device="cuda:0") for i in meta_args] + [i[1] for i in concrete_args]
    interp.propagate(*meta_tensors)
    codegen = AutoChunkCodeGen(meta_graph, max_memory=max_memory, print_mem=print_mem, print_progress=print_progress)
    chunks = codegen.chunk_infos

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

    # assert chunk in code
    code = graph.python_code("self").src
    if print_code:
        print(code)
    assert "chunk_result = None;  chunk_size = None;" in code

    # assert result
    inputs = [i[1] for i in meta_args] + [i[1] for i in concrete_args]
    model.cuda()
    with torch.no_grad():
        out_gm = gm(*inputs)
        out_model = model(*inputs)
    for out_gm_i, out_model_i in zip(out_gm, out_model):
        assert torch.allclose(out_gm_i, out_model_i,
                              atol=1e-4), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
                                  torch.abs(out_gm_i - out_model_i))

    return chunks

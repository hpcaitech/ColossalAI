from functools import partial

import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

try:
    from simple_evoformer import base_evoformer
    HAS_REPO = True
except:
    HAS_REPO = False

import colossalai
from colossalai.core import global_context as gpc
from colossalai.fx import ColoTracer, symbolic_trace
from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.codegen.activation_checkpoint_codegen import CODEGEN_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.utils import free_port

if CODEGEN_AVAILABLE and is_compatible_with_meta():
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor


def _test_fwd(model: torch.nn.Module, gm: ColoGraphModule, node, pair):
    with torch.no_grad():
        non_fx_out = model(node, pair)
        fx_out = gm(node, pair)

    assert torch.allclose(non_fx_out[0], fx_out[0],
                          atol=1e-4), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
                              torch.abs(non_fx_out[0] - fx_out[0]))
    assert torch.allclose(non_fx_out[1], fx_out[1],
                          atol=1e-4), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
                              torch.abs(non_fx_out[1] - fx_out[1]))


def _test_simple_evoformer_codegen(rank, msa_len, pair_len, max_memory):
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
    model = base_evoformer().cuda()
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()

    # meta info prop
    meta_graph = symbolic_trace(model,
                                meta_args={
                                    "node": node.to(torch.device("meta")),
                                    "pair": pair.to(torch.device("meta")),
                                })    # must use symbolic_trace
    interp = MetaInfoProp(meta_graph)
    interp.propagate(MetaTensor(node, fake_device="cuda:0"), MetaTensor(pair, fake_device="cuda:0"))
    codegen = AutoChunkCodeGen(meta_graph, max_memory=max_memory)

    # trace the module and replace codegen
    graph = ColoTracer().trace(
        model,
        meta_args={
            "node": node.to(torch.device("meta")),
            "pair": pair.to(torch.device("meta")),
        },
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # assert we have inserted chunk
    code = graph.python_code("self").src
    # print(code)
    assert "chunk_result = None;  chunk_size = None;" in code

    _test_fwd(model, gm, node, pair)
    gpc.destroy()


@pytest.mark.skipif(not (CODEGEN_AVAILABLE and is_compatible_with_meta() and HAS_REPO),
                    reason='torch version is lower than 1.12.0')
@pytest.mark.parametrize("max_memory", [None, 20, 25, 30])
@pytest.mark.parametrize("msa_len", [32])
@pytest.mark.parametrize("pair_len", [64])
def test_simple_evoformer_codegen(msa_len, pair_len, max_memory):
    run_func = partial(
        _test_simple_evoformer_codegen,
        msa_len=msa_len,
        pair_len=pair_len,
        max_memory=max_memory,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    _test_simple_evoformer_codegen(0, 32, 64, 25)

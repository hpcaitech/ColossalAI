from functools import partial

import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

try:
    from fastfold.model.nn.evoformer import ExtraMSABlock
    HAS_REPO = True
except:
    HAS_REPO = False

import colossalai
from colossalai.core import global_context as gpc
from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.codegen.activation_checkpoint_codegen import CODEGEN_AVAILABLE
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.utils import free_port

if CODEGEN_AVAILABLE and is_compatible_with_meta():
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor
    from colossalai.fx.tracer.experimental import ColoTracer, symbolic_trace


def _test_fwd(model: torch.nn.Module, gm: ColoGraphModule, node, pair, node_mask, pair_mask):
    # for memory test
    # model = model.cuda()
    # torch.cuda.reset_peak_memory_stats()
    # now_mem = torch.cuda.memory_allocated() / 1024**2
    # with torch.no_grad():
    #     node1 = node.clone()
    #     pair1 = pair.clone()
    #     node_mask1 = node_mask.clone()
    #     pair_mask1 = pair_mask.clone()
    #     gm(node1, pair1, node_mask1, pair_mask1)
    # new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
    # print("autochunk max mem:%.2f"% (new_max_mem - now_mem))

    # test forward
    model = model.cuda()
    with torch.no_grad():
        non_fx_out = model(node, pair, node_mask, pair_mask)
        fx_out = gm(node, pair, node_mask, pair_mask)

    assert torch.allclose(non_fx_out[0], fx_out[0],
                          atol=1e-4), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
                              torch.abs(non_fx_out[0] - fx_out[0]))
    assert torch.allclose(non_fx_out[1], fx_out[1],
                          atol=1e-4), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
                              torch.abs(non_fx_out[1] - fx_out[1]))


def _build_openfold():
    model = ExtraMSABlock(
        c_m=256,
        c_z=128,
        c_hidden_msa_att=32,
        c_hidden_opm=32,
        c_hidden_mul=128,
        c_hidden_pair_att=32,
        no_heads_msa=8,
        no_heads_pair=4,
        transition_n=4,
        msa_dropout=0.15,
        pair_dropout=0.15,
        inf=1e4,
        eps=1e-4,
        ckpt=False,
        is_multimer=False,
    ).eval().cuda()
    return model


def _test_extramsa_codegen(rank, msa_len, pair_len, max_memory):
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
    model = _build_openfold()
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    node_mask = torch.randn(1, msa_len, pair_len).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()
    pair_mask = torch.randn(1, pair_len, pair_len).cuda()

    # trace the meta graph and setup codegen
    meta_graph = symbolic_trace(
        model,
        meta_args={
            "m": node.to(torch.device("meta")),
            "z": pair.to(torch.device("meta")),
            "msa_mask": node_mask.to(torch.device("meta")),
            "pair_mask": pair_mask.to(torch.device("meta")),
        },
        concrete_args={
            "chunk_size": None,
            "_chunk_logits": 1024,
        },
    )
    interp = MetaInfoProp(meta_graph)
    interp.propagate(
        MetaTensor(node, fake_device="cuda:0"),
        MetaTensor(pair, fake_device="cuda:0"),
        MetaTensor(node_mask, fake_device="cuda:0"),
        MetaTensor(pair_mask, fake_device="cuda:0"),
    )
    codegen = AutoChunkCodeGen(meta_graph, max_memory=max_memory, print_mem=False)

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(
        model,
        meta_args={
            "m": node.to(torch.device("meta")),
            "z": pair.to(torch.device("meta")),
            "msa_mask": node_mask.to(torch.device("meta")),
            "pair_mask": pair_mask.to(torch.device("meta")),
        },
        concrete_args={
            "chunk_size": None,
            "_chunk_logits": 1024,
        },
    )
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # assert we have inserted chunk
    code = graph.python_code("self").src
    # print(code)
    assert "chunk_result = None;  chunk_size = None;" in code

    _test_fwd(model, gm, node, pair, node_mask, pair_mask)
    gpc.destroy()


@pytest.mark.skipif(
    not (CODEGEN_AVAILABLE and is_compatible_with_meta() and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@pytest.mark.parametrize("max_memory", [None, 24, 28, 32])
@pytest.mark.parametrize("msa_len", [32])
@pytest.mark.parametrize("pair_len", [64])
def test_extramsa_codegen(msa_len, pair_len, max_memory):
    run_func = partial(
        _test_extramsa_codegen,
        msa_len=msa_len,
        pair_len=pair_len,
        max_memory=max_memory,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    _test_extramsa_codegen(0, 32, 64, None)

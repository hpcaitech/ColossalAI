from functools import partial

import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

try:
    from fastfold.model.nn.evoformer import EvoformerBlock
    HAS_REPO = True
except:
    HAS_REPO = False

from test_alphafold_utils import assert_codegen_run

import colossalai
from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.core import global_context as gpc
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.utils import free_port


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
    model = EvoformerBlock(
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
        is_multimer=False,
    ).eval().cuda()
    return model


def _test_evoformer_codegen(rank, msa_len, pair_len, max_memory):
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

    assert_codegen_run(model,
                       meta_args=[("m", node), ("z", pair), ("msa_mask", node_mask), ("pair_mask", pair_mask)],
                       concrete_args=[("chunk_size", None), ("_mask_trans", True)],
                       max_memory=max_memory)

    gpc.destroy()


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@pytest.mark.parametrize("max_memory", [None, 24, 28, 32])
@pytest.mark.parametrize("msa_len", [32])
@pytest.mark.parametrize("pair_len", [64])
def test_evoformer_codegen(msa_len, pair_len, max_memory):
    run_func = partial(
        _test_evoformer_codegen,
        msa_len=msa_len,
        pair_len=pair_len,
        max_memory=max_memory,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    _test_evoformer_codegen(0, 32, 64, 24)

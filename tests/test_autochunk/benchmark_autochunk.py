import time

import torch
import torch.fx

from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
from colossalai.fx import ColoTracer
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import MetaTensor
from tests.test_autochunk.evoformer.evoformer import evoformer_base
from tests.test_autochunk.openfold.evoformer import EvoformerBlock


def _benchmark_evoformer(model: torch.nn.Module, node, pair, title, chunk_size=None):
    torch.cuda.reset_peak_memory_stats()
    now_mem = torch.cuda.memory_allocated() / 1024**2

    loop = 3
    with torch.no_grad():
        for _ in range(loop // 2 + 1):
            if chunk_size:
                model(node, pair, chunk_size)
            else:
                model(node, pair)
        torch.cuda.synchronize()
        time1 = time.time()
        for _ in range(loop):
            if chunk_size:
                model(node, pair, chunk_size)
            else:
                model(node, pair)
        torch.cuda.synchronize()
        time2 = time.time()

    new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(
        "%s: time %.4fs, mem %dMB"
        % (title, (time2 - time1) / loop, new_max_mem - now_mem)
    )


def _build_autochunk(model, max_memory, node, pair):
    # trace the module and replace codegen
    graph = ColoTracer().trace(
        model,
        meta_args={
            "node": node.to(torch.device("meta")),
            "pair": pair.to(torch.device("meta")),
        },
    )

    gm_prop = torch.fx.symbolic_trace(model)  # must use symbolic_trace
    interp = MetaInfoProp(gm_prop)
    interp.propagate(
        MetaTensor(node, fake_device="cuda:0"), MetaTensor(pair, fake_device="cuda:0")
    )

    # now run it twice to get meta info in graph module, not necessary
    gm = torch.fx.GraphModule(model, graph)
    interp = MetaInfoProp(gm)
    interp.propagate(
        MetaTensor(node, fake_device="cuda:0"), MetaTensor(pair, fake_device="cuda:0")
    )

    # set code_gen
    codegen = AutoChunkCodeGen(gm_prop, max_memory, print_mem=False)
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph)
    gm.recompile()

    # print
    # code = graph.python_code("self").src
    # print(code)
    return gm


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
    ).cuda()
    return model


def benchmark_evoformer():
    # init data and model
    msa_len = 256
    pair_len = 256
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()
    model = evoformer_base().cuda()

    # build autochunk model
    # max_memory = 1000  # MB fit memory mode
    max_memory = None  # min memory mode
    autochunk = _build_autochunk(evoformer_base().cuda(), max_memory, node, pair)

    # build openfold
    chunk_size = 64
    openfold = _build_openfold()

    # benchmark
    _benchmark_evoformer(model, node, pair, "base")
    _benchmark_evoformer(openfold, node, pair, "openfold", chunk_size=chunk_size)
    _benchmark_evoformer(autochunk, node, pair, "autochunk")


if __name__ == "__main__":
    benchmark_evoformer()

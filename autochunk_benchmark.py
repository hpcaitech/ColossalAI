import time

import torch
import torch.fx

from chunk_codegen import ChunkCodeGen
from colossalai.fx import ColoTracer
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import MetaTensor
from evoformer.evoformer import evoformer_base


def _benchmark_evoformer(model: torch.nn.Module, node, pair, title):
    torch.cuda.reset_peak_memory_stats()
    now_mem = torch.cuda.memory_allocated() / 1024**2

    loop = 16
    with torch.no_grad():
        for _ in range(loop // 4):
            model(node, pair)
        torch.cuda.synchronize()
        time1 = time.time()
        for _ in range(loop):
            model(node, pair)
        torch.cuda.synchronize()
        time2 = time.time()

    new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(
        "%s: time %.4fs, mem %dMB"
        % (title, (time2 - time1) / loop, new_max_mem - now_mem)
    )


def benchmark_evoformer():
    # data
    msa_len = 300
    pair_len = 800
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()

    # build gm model
    max_memory = 3000  # MB
    model = evoformer_base().cuda()
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
    codegen = ChunkCodeGen(gm_prop, max_memory)
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph)
    gm.recompile()
    # print
    code = graph.python_code("self").src
    print(code)

    _benchmark_evoformer(gm, node, pair, "autochunk")
    _benchmark_evoformer(model, node, pair, "openfold")


if __name__ == "__main__":
    benchmark_evoformer()

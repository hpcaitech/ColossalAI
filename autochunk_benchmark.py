import copy
import torch
import torch.nn.functional as F
import pytest
import torch.fx
import torch.multiprocessing as mp
from torch.fx import GraphModule
from colossalai.fx import ColoTracer
import colossalai
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp, TensorMetadata
from colossalai.fx.profiler import MetaTensor
from evoformer.evoformer import evoformer_base
from chunk_codegen import ChunkCodeGen
import time


def _benchmark_evoformer(model: torch.nn.Module, node, pair):
    loop = 10
    with torch.no_grad():
        for _ in range(loop // 4):
            model(node, pair)
        torch.cuda.synchronize()
        time1 = time.time()
        for _ in range(loop):
            model(node, pair)
        torch.cuda.synchronize()
        time2 = time.time()
    return (time2 - time1) / loop


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

    time_gm = _benchmark_evoformer(gm, node, pair)
    print("gm %.4fs" % time_gm)
    time_openfold = _benchmark_evoformer(model, node, pair)
    print("openfold %.4fs" % time_openfold)


if __name__ == "__main__":
    benchmark_evoformer()

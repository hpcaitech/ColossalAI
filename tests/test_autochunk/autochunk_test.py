import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

import colossalai
from colossalai.autochunk.chunk_codegen import ChunkCodeGen
from colossalai.core import global_context as gpc
from colossalai.fx import ColoTracer
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import MetaTensor
from colossalai.utils import free_port
from tests.test_autochunk.evoformer.evoformer import evoformer_base


def _test_fwd(model: torch.nn.Module, gm: ColoGraphModule, node, pair):
    torch.cuda.reset_peak_memory_stats()
    now_mem = torch.cuda.memory_allocated() / 1024**2
    with torch.no_grad():
        node1 = node.clone()
        pair1 = pair.clone()
        gm(node1, pair1)
    new_now_mem = torch.cuda.memory_allocated() / 1024**2
    new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(
        "autochunk now mem:%.2f max mem:%.2f"
        % (new_now_mem - now_mem, new_max_mem - now_mem)
    )

    # test forward
    with torch.no_grad():
        non_fx_out = model(node, pair)
        fx_out = gm(node, pair)

    assert torch.allclose(
        non_fx_out[0], fx_out[0], atol=1e-4
    ), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
        torch.abs(non_fx_out[0] - fx_out[0])
    )
    assert torch.allclose(
        non_fx_out[1], fx_out[1], atol=1e-4
    ), "fx_out doesn't comply with original output, diff is %.2e" % torch.mean(
        torch.abs(non_fx_out[1] - fx_out[1])
    )


def _run_offload_codegen(rank):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currectly
    colossalai.launch(
        config={},
        rank=rank,
        world_size=1,
        host="localhost",
        port=free_port(),
        backend="nccl",
    )

    # build model and input
    model = evoformer_base().cuda()
    node = torch.randn(1, 100, 300, 256).cuda()
    pair = torch.randn(1, 300, 300, 128).cuda()

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

    codegen = ChunkCodeGen(gm_prop)
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph)
    gm.recompile()

    # assert we have all the components
    # code = graph.python_code("self").src
    # print(code)

    _test_fwd(model, gm, node, pair)
    gpc.destroy()


def test_autochunk():
    mp.spawn(_run_offload_codegen, nprocs=1)


if __name__ == "__main__":
    _run_offload_codegen(0)

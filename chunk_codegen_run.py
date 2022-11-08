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
with_codegen = True


def _is_all_gradient_close(m: torch.nn.Module, gm: GraphModule) -> bool:
    for m_p, gm_p in zip(m.parameters(), gm.parameters()):
        if m_p.grad is not None and not torch.allclose(m_p.grad, gm_p.grad):
            return False
    return True


def _is_all_param_close(m: torch.nn.Module, gm: GraphModule) -> bool:
    for m_p, gm_p in zip(m.parameters(), gm.parameters()):
        if m_p.grad is not None and not torch.allclose(m_p.data, gm_p.data):
            return False
    return True


def _test_fwd_and_bwd(model: torch.nn.Module, gm: ColoGraphModule, node, pair):
    # now_mem = torch.cuda.memory_allocated() / 1024**2
    # max_mem = torch.cuda.max_memory_allocated() / 1024**2
    # print("now:%.2f max:%.2f" %(torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2))
    # with torch.no_grad():
    #     fx_out = gm(node, pair)
    # new_now_mem = torch.cuda.memory_allocated() / 1024**2
    # new_max_mem = torch.cuda.max_memory_allocated() / 1024**2
    # print("now:%.2f max:%.2f" %(new_now_mem - now_mem, new_max_mem - max_mem))
    
    # test forward
    with torch.no_grad():
        non_fx_out = model(node, pair)
        fx_out = gm(node, pair)
    assert torch.equal(non_fx_out[0], fx_out[0]), "fx_out doesn't comply with original output"
    assert torch.equal(non_fx_out[1], fx_out[1]), "fx_out doesn't comply with original output"

    # test barckward
    # loss0 = non_fx_out[0].sum() + non_fx_out[1].sum()
    # loss0.backward()
    # loss1 = fx_out[0].sum() + fx_out[1].sum()
    # loss1.backward()
    # assert _is_all_param_close(model, gm)
    # assert _is_all_gradient_close(model, gm), "gm doesn't have the same gradient as original one"


def _run_offload_codegen(rank):
    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currectly
    colossalai.launch(config={}, rank=rank, world_size=1, host='localhost', port=free_port(), backend='nccl')

    # build model and input
    model = evoformer_base().cuda()
    node = torch.randn(1, 16, 32, 256).cuda()
    pair = torch.randn(1, 32, 32, 128).cuda()

    # trace the module and replace codegen
    graph = ColoTracer().trace(model, meta_args={'node': node.to(torch.device('meta')), 'pair': pair.to(torch.device('meta'))})
    gm_prop = torch.fx.symbolic_trace(model) # must use symbolic_trace
    interp = MetaInfoProp(gm_prop) 
    interp.propagate(MetaTensor(node, fake_device='cuda:0'), MetaTensor(pair, fake_device='cuda:0'))

    # now run it twice to get meta info in graph module, not necessary
    gm = torch.fx.GraphModule(model, graph)
    interp = MetaInfoProp(gm)
    interp.propagate(MetaTensor(node, fake_device='cuda:0'), MetaTensor(pair, fake_device='cuda:0'))

    codegen = ChunkCodeGen(gm_prop)
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph)
    gm.recompile()

    # assert we have all the components
    code = graph.python_code("self").src
    print(code)

    _test_fwd_and_bwd(model, gm, node, pair)
    gpc.destroy()


@pytest.mark.skipif(not with_codegen, reason='torch version is lower than 1.12.0')
def test_act_ckpt_codegen():
    mp.spawn(_run_offload_codegen, nprocs=1)


if __name__ == "__main__":
    _run_offload_codegen(0)

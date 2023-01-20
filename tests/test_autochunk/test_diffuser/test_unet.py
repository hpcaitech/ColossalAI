from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

try:
    import diffusers
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

BATCH_SIZE = 2
SEQ_LENGTH = 5
HEIGHT = 224
WIDTH = 224
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 7, WIDTH // 7)


def _test_unet(rank, m, shape, timestep, max_memory):
    # launch colossalai
    colossalai.launch(
        config={},
        rank=rank,
        world_size=1,
        host="localhost",
        port=free_port(),
        backend="nccl",
    )

    model = m()
    sample = torch.randn(shape)

    # trace the meta graph and setup codegen
    meta_graph = symbolic_trace(model, meta_args={'sample': sample}, concrete_args=dict(timestep=timestep))
    interp = MetaInfoProp(meta_graph)
    interp.propagate(MetaTensor(sample, fake_device="cuda:0"), timestep)
    codegen = AutoChunkCodeGen(meta_graph, max_memory=max_memory, print_mem=False)

    # trace and recompile
    # MetaInfoProp requires symbolic_trace but CodeGen requires ColoTracer
    graph = ColoTracer().trace(model, meta_args={'sample': sample}, concrete_args=dict(timestep=timestep))
    graph.set_codegen(codegen)
    gm = ColoGraphModule(model, graph, ckpt_codegen=False)
    gm.recompile()

    # assert we have inserted chunk
    code = graph.python_code("self").src
    # print(code)
    assert "chunk_result = None;  chunk_size = None;" in code

    model.eval()
    gm.eval()

    with torch.no_grad():
        fx_out = gm(sample, timestep)
        non_fx_out = model(sample, timestep)
    assert torch.allclose(
        fx_out['sample'],
        non_fx_out['sample']), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


@pytest.mark.parametrize("m", [diffusers.UNet2DModel])
@pytest.mark.parametrize("shape", [LATENTS_SHAPE])
@pytest.mark.parametrize("timestep", [1])
@pytest.mark.parametrize("max_memory", [64])
def test_unet(m, shape, timestep, max_memory):
    run_func = partial(
        _test_unet,
        m=m,
        shape=shape,
        timestep=timestep,
        max_memory=max_memory,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    _test_unet(0, diffusers.UNet2DModel, LATENTS_SHAPE, 1, 64)

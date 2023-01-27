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
from colossalai.fx import symbolic_trace
from colossalai.fx._compatibility import is_compatible_with_meta
from colossalai.fx.codegen.activation_checkpoint_codegen import CODEGEN_AVAILABLE
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.utils import free_port

if CODEGEN_AVAILABLE and is_compatible_with_meta():
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen
    from colossalai.fx.profiler import MetaTensor


def assert_chunk_infos(chunk_infos, max_memory, msa_len, pair_len):
    found_regions = [i["region"] for i in chunk_infos]

    if msa_len == 32 and pair_len == 64:
        if max_memory is None:
            target_regions = [(142, 154), (366, 373), (234, 283), (302, 351), (127, 134), (211, 228), (174, 191),
                              (161, 166), (198, 203), (7, 57)]
        elif max_memory == 20:
            target_regions = [(142, 154), (369, 373), (235, 269), (303, 351), (130, 131)]
        elif max_memory == 25:
            target_regions = [(144, 154), (369, 370)]
        elif max_memory == 30:
            target_regions = [(144, 154)]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    assert found_regions == target_regions, "found regions %s doesn't equal target regions %s" % (
        str(found_regions),
        str(target_regions),
    )


def _test_simple_evoformer_search(rank, msa_len, pair_len, max_memory):
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

    meta_graph = symbolic_trace(model,
                                meta_args={
                                    "node": node.to(torch.device("meta")),
                                    "pair": pair.to(torch.device("meta")),
                                })    # must use symbolic_trace
    interp = MetaInfoProp(meta_graph)
    interp.propagate(MetaTensor(node, fake_device="cuda:0"), MetaTensor(pair, fake_device="cuda:0"))
    codegen = AutoChunkCodeGen(meta_graph, max_memory=max_memory)
    chunk_infos = codegen.chunk_infos
    assert_chunk_infos(chunk_infos, max_memory, msa_len, pair_len)

    gpc.destroy()


@pytest.mark.skipif(not (CODEGEN_AVAILABLE and is_compatible_with_meta() and HAS_REPO),
                    reason="torch version is lower than 1.12.0")
@pytest.mark.parametrize("max_memory", [None, 20, 25, 30])
@pytest.mark.parametrize("msa_len", [32])
@pytest.mark.parametrize("pair_len", [64])
def test_simple_evoformer_search(msa_len, pair_len, max_memory):
    run_func = partial(
        _test_simple_evoformer_search,
        msa_len=msa_len,
        pair_len=pair_len,
        max_memory=max_memory,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    _test_simple_evoformer_search(0, 32, 64, 20)

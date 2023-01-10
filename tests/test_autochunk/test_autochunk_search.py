from functools import partial

import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

import colossalai
from colossalai.core import global_context as gpc
from colossalai.fx.codegen.activation_checkpoint_codegen import CODEGEN_AVAILABLE
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import MetaTensor
from colossalai.utils import free_port
from tests.test_autochunk.evoformer.evoformer import evoformer_base

if CODEGEN_AVAILABLE:
    from colossalai.autochunk.autochunk_codegen import AutoChunkCodeGen


def assert_chunk_infos(chunk_infos, max_memory, msa_len, pair_len):
    found_regions = [i["region"] for i in chunk_infos]

    if msa_len == 32 and pair_len == 64:
        if max_memory is None:
            target_regions = [(142, 154), (366, 373), (233, 283), (301, 351), (127, 134), (204, 228), (167, 191), (161, 166), (198, 203), (6, 69)]
        elif max_memory == 20:
            target_regions = [(142, 154), (369, 373), (233, 269), (301, 351)]
        elif max_memory == 25:
            target_regions = [(144, 154), (369, 370)]
        elif max_memory == 30:
            target_regions = [(144, 154)]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    assert len(found_regions) == len(
        target_regions
    ), "len of found regions %s doesn't equal len of target regions %s" % (
        str(found_regions),
        str(target_regions),
    )
    for region in target_regions:
        assert (
            region in found_regions
        ), "region:%s not in found regions for msa:%d, pair:%d, maxmem:%d" % (
            str(region),
            msa_len,
            pair_len,
            max_memory,
        )
    for region in found_regions:
        assert (
            region in target_regions
        ), "region:%s should not be found for msa:%d, pair:%d, maxmem:%d" % (
            str(region),
            msa_len,
            pair_len,
            max_memory,
        )


def _test_autochunk_search(rank, msa_len, pair_len, max_memory):
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
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()

    gm_prop = torch.fx.symbolic_trace(model)  # must use symbolic_trace
    interp = MetaInfoProp(gm_prop)
    interp.propagate(
        MetaTensor(node, fake_device="cuda:0"), MetaTensor(pair, fake_device="cuda:0")
    )

    codegen = AutoChunkCodeGen(gm_prop, max_memory=max_memory)
    chunk_infos = codegen.chunk_infos
    assert_chunk_infos(chunk_infos, max_memory, msa_len, pair_len)

    gpc.destroy()


@pytest.mark.skipif(not CODEGEN_AVAILABLE, reason="torch version is lower than 1.12.0")
@pytest.mark.parametrize("max_memory", [None, 20, 25, 30])
@pytest.mark.parametrize("msa_len", [32])
@pytest.mark.parametrize("pair_len", [64])
def test_autochunk_search(msa_len, pair_len, max_memory):
    run_func = partial(
        _test_autochunk_search,
        msa_len=msa_len,
        pair_len=pair_len,
        max_memory=max_memory,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    _test_autochunk_search(0, 32, 64, 20)

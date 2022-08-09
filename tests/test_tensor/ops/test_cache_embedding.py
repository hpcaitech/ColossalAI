import pytest
from functools import partial
import torch
import torch.multiprocessing as mp
import numpy as np

from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use

from colossalai.nn._ops.cache_embedding import CachedParamMgr

NUM_EMBED, EMBED_DIM = 100, 8
BATCH_SIZE = 8


def test_cachemgr():
    model = torch.nn.EmbeddingBag(10000, 128)
    # 10 chunks, 5 in cuda
    mgr = CachedParamMgr(model.weight, 5)
    assert mgr.cuda_row_num == 5

    mgr._admit(1)
    assert not mgr._chunk_in_cuda(2)
    assert mgr._chunk_in_cuda(1)

    # print(mgr.cached_chunk_table)
    mgr._admit(8)

    # now 3 chunk is available
    assert mgr.cuda_available_chunk_num == 3

    mgr._evict()
    assert mgr.cuda_available_chunk_num == 4

    mgr._prepare_rows_on_cuda(torch.tensor([9, 6, 5], dtype=torch.long, device=0))
    mgr._prepare_rows_on_cuda(torch.tensor([3, 4, 5], dtype=torch.long, device=0))
    # print(mgr.cached_chunk_table)
    # mgr.print_comm_stats()

    mgr.flush()
    assert mgr.cuda_available_chunk_num == 5


def test_reorder_with_freq():
    num_embed = 100
    chunk_size = 1
    num_chunk = 5

    idx_map = np.random.randint(10000, size=(num_embed,))
    sorted_idx = np.flipud(np.argsort(idx_map)).tolist()
    chunkid, offset_in_chunk = [], []
    for i in range(num_embed):
        idx = sorted_idx.index(i)
        chunkid.append(idx // chunk_size)
        offset_in_chunk.append(idx % chunk_size)

    chunkid = torch.tensor(chunkid, dtype=torch.long, device=torch.cuda.current_device())
    offset_in_chunk = torch.tensor(offset_in_chunk, dtype=torch.long, device=torch.cuda.current_device())

    weight = torch.rand(num_embed, 2)
    mgr = CachedParamMgr(weight, num_chunk)

    mgr.reorder(idx_map)

    indices = mgr.idx_map.index_select(0, torch.arange(num_embed, dtype=torch.long, device=torch.cuda.current_device()))
    mgr_chunk_id = torch.div(indices, chunk_size, rounding_mode='floor')
    mgr_offsets = torch.remainder(indices, chunk_size)
    assert torch.allclose(chunkid, mgr_chunk_id), f"chunk id: {chunkid}, mgr: {mgr_chunk_id}"
    assert torch.allclose(offset_in_chunk, mgr_offsets), \
        f"offset in chunk: {offset_in_chunk}, mgr: {mgr_offsets}"


if __name__ == '__main__':
    # test_freq_aware_embed()
    # test_chunkmgr_admit()
    pass

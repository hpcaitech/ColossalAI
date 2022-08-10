import pytest
from functools import partial
import torch
import torch.multiprocessing as mp
import numpy as np
import random

import colossalai
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.tensor import ColoParameter, ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.nn._ops.cache_embedding import CachedParamMgr, FreqAwareEmbeddingBag, ParallelFreqAwareEmbeddingBag

NUM_EMBED, EMBED_DIM = 10, 8
BATCH_SIZE = 8


def set_seed(seed):
    """
    To achieve reproducible results, it's necessary to fix random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthesize_1d_sparse_feature(
    batch_size,
    num_embed,
    device,
):
    indices_in_batch = batch_size * 2
    indices = torch.randint(low=0, high=num_embed, size=(indices_in_batch,), device=device, dtype=torch.long)
    offsets = torch.from_numpy(
        np.array([
            0, *np.sort(np.random.randint(low=0, high=indices_in_batch, size=(indices_in_batch - 1,))), indices_in_batch
        ])).to(device).long()
    return indices, offsets


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


def test_freq_aware_embed():
    device = torch.device('cuda', 0)
    model = FreqAwareEmbeddingBag(
        NUM_EMBED,
        EMBED_DIM,
        mode='mean',
        include_last_offset=True,
    ).to(device)
    model.preprocess(cuda_row_num=BATCH_SIZE * 2, ids_freq_mapping=None)

    assert model.weight.shape[0] == NUM_EMBED
    ref_model = torch.nn.EmbeddingBag.from_pretrained(model.weight.detach().to(device),
                                                      mode='mean',
                                                      include_last_offset=True,
                                                      freeze=False)

    assert torch.allclose(ref_model.weight.detach(), model.weight.detach().to(device))

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for i in range(5):
        indices, offsets = synthesize_1d_sparse_feature(BATCH_SIZE, NUM_EMBED, device)
        res = model(indices, offsets)
        ref_res = ref_model(indices, offsets)
        assert torch.allclose(res, ref_res), f"model result: {res}, reference: {ref_res}"

        grad = torch.rand_like(res)
        # comparing gradient here is nontrivial
        res.backward(grad)
        ref_res.backward(grad)
        optimizer.step()
        optimizer.zero_grad()

        ref_optimizer.step()
        ref_optimizer.zero_grad()

    model.cache_weight_mgr.flush()
    model_weight = model.weight.detach().to(device)
    ref_weight = ref_model.weight.detach()
    assert torch.allclose(model_weight, ref_weight), \
        f"model weight: {model_weight[10:18, :8]}, reference: {ref_weight[10:18, :8]}"


def gather_tensor(tensor, rank, world_size):
    gather_list = []
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    torch.distributed.gather(tensor, gather_list, dst=0)
    return gather_list


def run_parallel_freq_aware_embed(rank, world_size):
    device = torch.device('cuda', torch.cuda.current_device())

    num_embed = 100
    embed_dim = 16
    batch_size = 4

    set_seed(4321)
    weight = torch.rand(num_embed, embed_dim)
    coloweight = ColoParameter(weight.clone().detach().cpu(), requires_grad=False)

    # initialize the tensor spec for the embedding weight parameter,
    # which is an ColoParameter.
    coloweight.process_group = ProcessGroup(tp_degree=world_size)
    coloweight.set_tensor_spec(ShardSpec(dims=[-1], num_partitions=[world_size]), ComputeSpec(ComputePattern.TP1D))

    model = ParallelFreqAwareEmbeddingBag.from_pretrained(coloweight,
                                                          include_last_offset=True,
                                                          freeze=False,
                                                          cuda_row_num=batch_size * 2)

    assert model.cache_weight_mgr.cpu_weight.device.type == 'cpu'
    assert model.cache_weight_mgr.cuda_cached_weight.requires_grad
    weight_in_rank = torch.tensor_split(weight, world_size, -1)[rank]
    assert torch.allclose(
        weight_in_rank,
        model.cache_weight_mgr.cpu_weight.detach()), f"{weight_in_rank - model.cache_weight_mgr.cpu_weight}"

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if rank == 0:
        ref_model = torch.nn.EmbeddingBag.from_pretrained(weight.detach().clone(),
                                                          include_last_offset=True,
                                                          freeze=False).to(device)
        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    set_seed(4321)
    for i in range(5):
        indices, offsets = synthesize_1d_sparse_feature(batch_size, num_embed, device)
        res = model(indices, offsets)

        grad = torch.rand(batch_size * 2, embed_dim, dtype=res.dtype, device=res.device)
        grad_in_rank = torch.tensor_split(grad, world_size, 0)[rank]
        res.backward(grad_in_rank)

        optimizer.step()
        optimizer.zero_grad()

        res_list = gather_tensor(res.detach(), rank, world_size)

        if rank == 0:
            ref_res = ref_model(indices, offsets)
            recover_res = torch.cat(res_list, dim=0)

            assert torch.allclose(ref_res, recover_res)

            ref_res.backward(grad)
            ref_optimizer.step()
            ref_optimizer.zero_grad()

    model.cache_weight_mgr.flush()
    weight_list = gather_tensor(model.cache_weight_mgr.cpu_weight.detach().cuda(), rank, world_size)
    if rank == 0:
        recover_weight = torch.cat(weight_list, dim=1)
        assert torch.allclose(recover_weight, ref_model.weight.detach()), f"{recover_weight - ref_model.weight}"


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_parallel_freq_aware_embed(rank, world_size)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_parallel_freq_aware_embed(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    # test_freq_aware_embed()
    # test_chunkmgr_admit()
    test_parallel_freq_aware_embed(2)

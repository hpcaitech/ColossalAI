import random
from typing import List

import numpy as np
import pytest
import torch

import colossalai
from colossalai.legacy.nn.parallel.layers import (
    CachedEmbeddingBag,
    CachedParamMgr,
    EvictionStrategy,
    ParallelCachedEmbeddingBag,
    ParallelCachedEmbeddingBagTablewise,
    TablewiseEmbeddingBagConfig,
)
from colossalai.legacy.tensor import ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.tensor import ColoTensor
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

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
    offsets = (
        torch.from_numpy(
            np.array(
                [
                    0,
                    *np.sort(np.random.randint(low=0, high=indices_in_batch, size=(indices_in_batch - 1,))),
                    indices_in_batch,
                ]
            )
        )
        .to(device)
        .long()
    )
    return indices, offsets


@pytest.mark.skip
@clear_cache_before_run()
def test_cachemgr():
    model = torch.nn.EmbeddingBag(10000, 128)
    # 10 chunks, 5 in cuda
    mgr = CachedParamMgr(model.weight.detach(), 5)
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


@clear_cache_before_run()
def test_reorder_with_freq():
    num_embed = 100
    chunk_size = 1
    num_chunk = 5

    idx_map = torch.randint(10000, size=(num_embed,))
    sorted_idx = torch.argsort(idx_map, descending=True).tolist()
    chunkid, offset_in_chunk = [], []
    for i in range(num_embed):
        idx = sorted_idx.index(i)
        chunkid.append(idx // chunk_size)
        offset_in_chunk.append(idx % chunk_size)

    dev = torch.device("cuda")
    chunkid = torch.tensor(chunkid, dtype=torch.long, device=dev)
    offset_in_chunk = torch.tensor(offset_in_chunk, dtype=torch.long, device=dev)

    weight = torch.rand(num_embed, 2)
    mgr = CachedParamMgr(weight, num_chunk)

    mgr.reorder(idx_map)

    indices = mgr.idx_map.index_select(0, torch.arange(num_embed, dtype=torch.long, device=dev))
    mgr_chunk_id = torch.div(indices, chunk_size, rounding_mode="floor")
    mgr_offsets = torch.remainder(indices, chunk_size)
    assert torch.allclose(chunkid, mgr_chunk_id), f"chunk id: {chunkid}, mgr: {mgr_chunk_id}"
    assert torch.allclose(offset_in_chunk, mgr_offsets), f"offset in chunk: {offset_in_chunk}, mgr: {mgr_offsets}"


@clear_cache_before_run()
@parameterize("use_LFU", [True, False])
def test_freq_aware_embed(use_LFU: bool):
    device = torch.device("cuda", 0)
    evict_strategy = EvictionStrategy.LFU if use_LFU else EvictionStrategy.DATASET
    model = CachedEmbeddingBag(
        NUM_EMBED,
        EMBED_DIM,
        mode="mean",
        include_last_offset=True,
        cache_ratio=min(BATCH_SIZE * 2 / NUM_EMBED, 1.0),
        ids_freq_mapping=None,
        evict_strategy=evict_strategy,
    ).to(device)

    assert model.weight.shape[0] == NUM_EMBED
    ref_model = torch.nn.EmbeddingBag.from_pretrained(
        model.weight.detach().to(device), mode="mean", include_last_offset=True, freeze=False
    )

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
    assert torch.allclose(
        model_weight, ref_weight
    ), f"model weight: {model_weight[10:18, :8]}, reference: {ref_weight[10:18, :8]}"


@clear_cache_before_run()
@parameterize("init_freq", [True, False])
def test_lfu_strategy(init_freq: bool):
    # minimal test to check behavior
    Bag = CachedEmbeddingBag(
        5,
        5,
        cache_ratio=3 / 5,
        buffer_size=0,
        pin_weight=True,
        ids_freq_mapping=[4, 2, 1, 3, 1] if init_freq else None,
        warmup_ratio=1.0,
        evict_strategy=EvictionStrategy.LFU,
    )

    # print('cached_idx_map: ', Bag.cache_weight_mgr.cached_idx_map)
    offsets = torch.tensor([0], device="cuda:0")

    # prepare frequency learning info:
    Bag.forward(torch.tensor([2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)

    # check strategy
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([0, 1, 2], device="cuda:0"), offsets)
    Bag.forward(torch.tensor([3], device="cuda:0"), offsets)  # miss, evict 1
    Bag.forward(torch.tensor([2], device="cuda:0"), offsets)  # hit
    Bag.forward(torch.tensor([4], device="cuda:0"), offsets)  # miss, evict 3
    Bag.forward(torch.tensor([2], device="cuda:0"), offsets)  # hit
    Bag.forward(torch.tensor([0], device="cuda:0"), offsets)  # hit

    assert torch.allclose(
        torch.Tensor(Bag.cache_weight_mgr.num_hits_history[-6:]), torch.Tensor([3, 0, 1, 0, 1, 1])
    ), "LFU strategy behavior failed"


def gather_tensor(tensor, rank, world_size):
    gather_list = []
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    torch.distributed.gather(tensor, gather_list, dst=0)
    return gather_list


def run_parallel_freq_aware_embed_tablewise(rank, world_size):
    if world_size != 2:
        return
    device = torch.device("cuda", torch.cuda.current_device())

    # initialize weight
    # 3 feature tables. idx: 0~5, 6~10, 11~17
    weight_tables = torch.rand(18, 5)
    weight_table1 = weight_tables[0:6]
    weight_table2 = weight_tables[6:11]
    weight_table3 = weight_tables[11:18]
    embedding_bag_config_list: List[TablewiseEmbeddingBagConfig] = []
    embedding_bag_config_list.append(
        TablewiseEmbeddingBagConfig(
            num_embeddings=6, cuda_row_num=4, assigned_rank=0, initial_weight=weight_table1.clone().detach().cpu()
        )
    )
    embedding_bag_config_list.append(
        TablewiseEmbeddingBagConfig(
            num_embeddings=5, cuda_row_num=4, assigned_rank=0, initial_weight=weight_table2.clone().detach().cpu()
        )
    )
    embedding_bag_config_list.append(
        TablewiseEmbeddingBagConfig(
            num_embeddings=7, cuda_row_num=4, assigned_rank=1, initial_weight=weight_table3.clone().detach().cpu()
        )
    )
    if rank == 0:
        _weight = torch.cat([weight_table1, weight_table2], 0)
    else:
        _weight = weight_table3
    model = ParallelCachedEmbeddingBagTablewise(
        embedding_bag_config_list,
        embedding_dim=5,
        _weight=_weight,
        include_last_offset=True,
        cache_ratio=0.5,
        buffer_size=0,
        evict_strategy=EvictionStrategy.LFU,
    )
    # explain
    """
    batch       feature 1       feature 2       feature 3
    input0      [1,2,3]         [6,7]           []
    input1      []              [9]             [13,15]
    input2      [1,5]           [6,8]           [11]
                  ↑               ↑               ↑
                rank 0          rank 0          rank 1
    in KJT format
    """
    res = model(
        torch.tensor([1, 2, 3, 1, 5, 6, 7, 9, 6, 8, 13, 15, 11], device=device),
        torch.tensor([0, 3, 3, 5, 7, 8, 10, 10, 12, 13], device=device),
        already_split_along_rank=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    rand_grad = torch.rand(3, 5 * 3, dtype=res.dtype, device=res.device)
    if rank == 0:
        fake_grad = rand_grad[0:2]
    else:
        fake_grad = rand_grad[2:]
    res.backward(fake_grad)
    optimizer.step()
    optimizer.zero_grad()

    # check correctness
    if rank == 0:
        ref_model = torch.nn.EmbeddingBag.from_pretrained(
            weight_tables.detach().clone(), include_last_offset=True, freeze=False
        ).to(device)
        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-2)
        ref_fake_grad = torch.cat(rand_grad.split(5, 1), 0)
        ref_res = ref_model(
            torch.tensor([1, 2, 3, 1, 5, 6, 7, 9, 6, 8, 13, 15, 11], device=device),
            torch.tensor([0, 3, 3, 5, 7, 8, 10, 10, 12, 13], device=device),
        )
        ref_res.backward(ref_fake_grad)
        ref_optimizer.step()
        ref_optimizer.zero_grad()

        model.cache_weight_mgr.flush()
        recover_weight = model.cache_weight_mgr.weight.to(device)
        ref_weight = ref_model.weight.detach()[:11]
        assert torch.allclose(recover_weight, ref_weight), f"{recover_weight - ref_weight}"


def run_parallel_freq_aware_embed_columnwise(rank, world_size):
    device = torch.device("cuda", torch.cuda.current_device())

    num_embed = 100
    embed_dim = 16
    batch_size = 4

    set_seed(4321)
    weight = torch.rand(num_embed, embed_dim)
    coloweight = ColoTensor(weight.clone().detach().cpu(), spec=None)

    # initialize the tensor spec for the embedding weight parameter,
    # which is an ColoParameter.
    coloweight.set_process_group(ProcessGroup(tp_degree=world_size))
    coloweight.set_tensor_spec(ShardSpec(dims=[-1], num_partitions=[world_size]), ComputeSpec(ComputePattern.TP1D))

    model = ParallelCachedEmbeddingBag.from_pretrained(
        coloweight,
        include_last_offset=True,
        freeze=False,
        cache_ratio=batch_size * 2 / num_embed,
    )

    assert model.cache_weight_mgr.weight.device.type == "cpu"
    assert model.cache_weight_mgr.cuda_cached_weight.requires_grad
    weight_in_rank = torch.tensor_split(weight, world_size, -1)[rank]
    print(f"model weight: {model.cache_weight_mgr.weight.shape}, ref weight: {weight_in_rank.shape}")
    assert torch.allclose(
        weight_in_rank, model.cache_weight_mgr.weight.detach()
    ), f"{weight_in_rank - model.cache_weight_mgr.weight}"

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if rank == 0:
        ref_model = torch.nn.EmbeddingBag.from_pretrained(
            weight.detach().clone(), include_last_offset=True, freeze=False
        ).to(device)
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
    weight_list = gather_tensor(model.cache_weight_mgr.weight.detach().cuda(), rank, world_size)
    if rank == 0:
        recover_weight = torch.cat(weight_list, dim=1)
        assert torch.allclose(recover_weight, ref_model.weight.detach()), f"{recover_weight - ref_model.weight}"


def run_dist(rank, world_size, port):
    colossalai.legacy.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # run_parallel_freq_aware_embed_columnwise(rank, world_size)
    run_parallel_freq_aware_embed_tablewise(rank, world_size)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_parallel_freq_aware_embed(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    # test_freq_aware_embed(True)
    test_parallel_freq_aware_embed(2)
    # test_lfu_strategy(False)

import torch
from colossalai.utils.checkpoint_io.meta import ParamRedistMeta
from colossalai.utils.checkpoint_io.distributed import flatten_zero_param, split_tp_param, unmerge_param


def test_flatten_zero_param_even() -> None:
    redist_meta = ParamRedistMeta(4, 1, zero_start_dp_rank=0, zero_offsets=[0, 4, 8, 12])
    orig_tensor = torch.rand(4, 4)
    tensors = list(orig_tensor.reshape(-1).chunk(4))
    flat_tensors = flatten_zero_param(orig_tensor, redist_meta)
    assert len(tensors) == len(flat_tensors)
    for t, st in zip(tensors, flat_tensors):
        assert torch.equal(t, st)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(unmerged_tensors) == 1
    unmerged_tensors = unmerged_tensors[0]
    assert len(tensors) == len(unmerged_tensors)
    for t, tl in zip(tensors, unmerged_tensors):
        assert torch.equal(t, tl)


def test_flatten_zero_param_uneven() -> None:
    redist_meta = ParamRedistMeta(4, 1, zero_start_dp_rank=1, zero_offsets=[0, 13])
    orig_tensor = torch.rand(4, 4)
    tensors = list(orig_tensor.reshape(-1).split([13, 3]))
    flat_tensors = flatten_zero_param(orig_tensor, redist_meta)
    assert flat_tensors[0].size(0) == 0 and flat_tensors[-1].size(0) == 0
    flat_tensors = flat_tensors[1:-1]
    assert len(tensors) == len(flat_tensors)
    for t, st in zip(tensors, flat_tensors):
        assert torch.equal(t, st)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(unmerged_tensors) == 1
    unmerged_tensors = unmerged_tensors[0]
    assert unmerged_tensors[0].size(0) == 0 and unmerged_tensors[-1].size(0) == 0
    unmerged_tensors = unmerged_tensors[1:-1]
    assert len(tensors) == len(unmerged_tensors)
    for t, tl in zip(tensors, unmerged_tensors):
        assert torch.equal(t, tl)


def test_split_tp_param_1d_row() -> None:
    redist_meta = ParamRedistMeta(1, 4, tp_shard_dims=[0], tp_num_parts=[4])
    orig_tensor = torch.rand(4, 4)
    tensors = [t.contiguous() for t in orig_tensor.chunk(4, 0)]
    split_tensors = split_tp_param(orig_tensor, redist_meta)
    assert len(tensors) == len(split_tensors)
    for t, st in zip(tensors, split_tensors):
        assert torch.equal(t, st)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(tensors) == len(unmerged_tensors)
    for t, tl in zip(tensors, unmerged_tensors):
        assert len(tl) == 1
        assert torch.equal(t, tl[0])


def test_split_tp_param_1d_col() -> None:
    redist_meta = ParamRedistMeta(1, 4, tp_shard_dims=[1], tp_num_parts=[4])
    orig_tensor = torch.rand(4, 4)
    tensors = [t.contiguous() for t in orig_tensor.chunk(4, 1)]
    split_tensors = split_tp_param(orig_tensor, redist_meta)
    assert len(tensors) == len(split_tensors)
    for t, st in zip(tensors, split_tensors):
        assert torch.equal(t, st)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(tensors) == len(unmerged_tensors)
    for t, tl in zip(tensors, unmerged_tensors):
        assert len(tl) == 1
        assert torch.equal(t, tl[0])


def test_split_tp_param_2d() -> None:
    redist_meta = ParamRedistMeta(1, 6, tp_shard_dims=[0, 1], tp_num_parts=[2, 3])
    orig_tensor = torch.rand(4, 6)
    tensors = [t.contiguous() for tl in orig_tensor.chunk(2, 0) for t in tl.chunk(3, 1)]
    split_tensors = split_tp_param(orig_tensor, redist_meta)
    assert len(tensors) == len(split_tensors)
    for t, st in zip(tensors, split_tensors):
        assert torch.equal(t, st)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(tensors) == len(unmerged_tensors)
    for t, tl in zip(tensors, unmerged_tensors):
        assert len(tl) == 1
        assert torch.equal(t, tl[0])


def test_split_tp_param_2d_reverse() -> None:
    redist_meta = ParamRedistMeta(1, 6, tp_shard_dims=[1, 0], tp_num_parts=[3, 2])
    orig_tensor = torch.rand(4, 6)
    tensors = [t.contiguous() for tl in orig_tensor.chunk(2, 0) for t in tl.chunk(3, 1)]
    split_tensors = split_tp_param(orig_tensor, redist_meta)
    assert len(tensors) == len(split_tensors)
    for t, st in zip(tensors, split_tensors):
        assert torch.equal(t, st)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(tensors) == len(unmerged_tensors)
    for t, tl in zip(tensors, unmerged_tensors):
        assert len(tl) == 1
        assert torch.equal(t, tl[0])


def test_unmerge_param_hybrid() -> None:
    redist_meta = ParamRedistMeta(2,
                                  6,
                                  tp_shard_dims=[1, 0],
                                  tp_num_parts=[3, 2],
                                  zero_start_dp_rank=0,
                                  zero_offsets=[0, 1])
    orig_tensor = torch.rand(4, 6)
    tensors = [
        chunk for tl in orig_tensor.chunk(2, 0) for t in tl.chunk(3, 1)
        for chunk in t.contiguous().reshape(-1).split([1, 3])
    ]
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(unmerged_tensors) == 6 and len(unmerged_tensors[0]) == 2
    for tp_rank in range(6):
        for dp_rank in range(2):
            assert torch.equal(tensors[tp_rank * 2 + dp_rank], unmerged_tensors[tp_rank][dp_rank])


def test_unmerge_param_dummy() -> None:
    redist_meta = ParamRedistMeta(1, 1)
    orig_tensor = torch.rand(4, 6)
    unmerged_tensors = unmerge_param(orig_tensor, redist_meta)
    assert len(unmerged_tensors) == 1 and len(unmerged_tensors[0]) == 1
    assert torch.equal(orig_tensor, unmerged_tensors[0][0])


if __name__ == '__main__':
    test_flatten_zero_param_even()
    test_flatten_zero_param_uneven()
    test_split_tp_param_1d_row()
    test_split_tp_param_1d_col()
    test_split_tp_param_2d()
    test_split_tp_param_2d_reverse()
    test_unmerge_param_hybrid()
    test_unmerge_param_dummy()

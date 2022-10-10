import torch
from colossalai.utils.checkpoint_io.meta import ParamDistMeta
from colossalai.utils.checkpoint_io.distributed import unflatten_zero_param, gather_tp_param, merge_param


def test_unflatten_zero_param_even() -> None:
    dist_metas = [ParamDistMeta(i, 4, 0, 1, zero_numel=16, zero_orig_shape=[4, 4]) for i in range(4)]
    orig_tensor = torch.rand(4, 4)
    tensors = list(orig_tensor.reshape(-1).chunk(4))
    unflattened_tensor = unflatten_zero_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, unflattened_tensor)
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_unflatten_zero_param_uneven() -> None:
    dist_metas = [ParamDistMeta(i, 4, 0, 1, zero_numel=16, zero_orig_shape=[4, 4]) for i in range(1, 3)]
    orig_tensor = torch.rand(4, 4)
    tensors = list(orig_tensor.reshape(-1).split([13, 3]))
    unflattened_tensor = unflatten_zero_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, unflattened_tensor)
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_gather_tp_param_1d_row() -> None:
    dist_metas = [ParamDistMeta(0, 1, i, 4, tp_shard_dims=[0], tp_num_parts=[4]) for i in range(4)]
    orig_tensor = torch.rand(4, 4)
    tensors = [t.contiguous() for t in orig_tensor.chunk(4, 0)]
    gathered_tensor = gather_tp_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, gathered_tensor)
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_gather_tp_param_1d_col() -> None:
    dist_metas = [ParamDistMeta(0, 1, i, 4, tp_shard_dims=[1], tp_num_parts=[4]) for i in range(4)]
    orig_tensor = torch.rand(4, 4)
    tensors = [t.contiguous() for t in orig_tensor.chunk(4, 1)]
    gathered_tensor = gather_tp_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, gathered_tensor)
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_gather_tp_param_2d() -> None:
    dist_metas = [ParamDistMeta(0, 1, i, 6, tp_shard_dims=[0, 1], tp_num_parts=[2, 3]) for i in range(6)]
    orig_tensor = torch.rand(4, 6)
    tensors = [t.contiguous() for tl in orig_tensor.chunk(2, 0) for t in tl.chunk(3, 1)]
    gathered_tensor = gather_tp_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, gathered_tensor)
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_gather_tp_param_2d_reverse() -> None:
    dist_metas = [ParamDistMeta(0, 1, i, 6, tp_shard_dims=[1, 0], tp_num_parts=[3, 2]) for i in range(6)]
    orig_tensor = torch.rand(4, 6)
    tensors = [t.contiguous() for tl in orig_tensor.chunk(2, 0) for t in tl.chunk(3, 1)]
    gathered_tensor = gather_tp_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, gathered_tensor)
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_merge_param_hybrid() -> None:
    dist_metas = [
        ParamDistMeta(i % 2,
                      2,
                      i // 2,
                      6,
                      tp_shard_dims=[1, 0],
                      tp_num_parts=[3, 2],
                      zero_numel=4,
                      zero_orig_shape=[2, 2]) for i in range(12)
    ]
    orig_tensor = torch.rand(4, 6)
    tensors = [
        chunk for tl in orig_tensor.chunk(2, 0) for t in tl.chunk(3, 1)
        for chunk in t.contiguous().reshape(-1).split([1, 3])
    ]
    merged_tensor = merge_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


def test_merge_param_dummy() -> None:
    dist_metas = [ParamDistMeta(0, 1, 0, 1)]
    orig_tensor = torch.rand(4, 6)
    merged_tensor = merge_param([orig_tensor], dist_metas)
    assert torch.equal(orig_tensor, merged_tensor)


if __name__ == '__main__':
    test_unflatten_zero_param_even()
    test_unflatten_zero_param_uneven()
    test_gather_tp_param_1d_row()
    test_gather_tp_param_1d_col()
    test_gather_tp_param_2d()
    test_gather_tp_param_2d_reverse()
    test_merge_param_hybrid()
    test_merge_param_dummy()

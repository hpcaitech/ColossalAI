import torch
from numpy import prod
from torch import Tensor
from typing import List, Optional, Tuple
from collections import defaultdict
from .meta import ParamDistMeta, ParamRedistMeta


def unflatten_zero_param(tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> Tensor:
    assert len(tensors) > 0 and len(dist_metas) > 0 and len(tensors) == len(dist_metas)
    for dist_meta in dist_metas[1:]:
        assert dist_meta.zero_meta == dist_metas[0].zero_meta, 'Expect all params have the same zero meta.'
    if not dist_metas[0].used_zero:
        # tensors are replicate
        return tensors[0]
    numel = dist_metas[0].zero_numel
    orig_shape = dist_metas[0].zero_orig_shape
    tensors = [t[1] for t in sorted(zip(dist_metas, tensors), key=lambda tp: tp[0].dp_rank)]
    assert numel == sum(t.numel() for t in tensors), 'Expect numel of all params is equal to zero_numel.'
    return torch.cat(tensors).reshape(orig_shape)


def gather_tp_param(tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> Tensor:
    assert len(tensors) > 0 and len(dist_metas) > 0 and len(tensors) == len(dist_metas)
    for dist_meta in dist_metas[1:]:
        assert dist_meta.tp_meta == dist_metas[0].tp_meta, 'Expect all params have the same tp meta.'
    for t in tensors[1:]:
        assert t.shape == tensors[0].shape, 'Expect all params have the same shape.'
    if not dist_metas[0].used_tp:
        # tensors are replicate
        return tensors[0]
    total_parts = prod(dist_meta.tp_num_parts)
    assert dist_meta.tp_world_size == total_parts, \
        f'Expect prod(tp_num_parts) == tp_world_size, got {total_parts} and {dist_meta.tp_world_size}.'
    shard_info = sorted(zip(dist_meta.tp_shard_dims, dist_meta.tp_num_parts), key=lambda t: t[0], reverse=True)
    for dim, num_parts in shard_info:
        buffer = []
        for start in range(0, len(tensors), num_parts):
            buffer.append(torch.cat(tensors[start:start + num_parts], dim))
        tensors = buffer
    assert len(tensors) == 1
    return tensors[0]


def validate_parallel_info(dist_metas: List[ParamDistMeta]) -> None:
    assert len(dist_metas) > 0
    # check world size
    for dist_meta in dist_metas[1:]:
        assert dist_meta.dp_world_size == dist_metas[
            0].dp_world_size, 'Expect all dist meta have the same dp_world_size'
        assert dist_meta.tp_world_size == dist_metas[
            0].tp_world_size, 'Expect all dist meta have the same tp_world_size'


def deduplicate_params(tensors: List[Tensor],
                       dist_metas: List[ParamDistMeta]) -> Tuple[List[Tensor], List[ParamDistMeta]]:
    unique_dist_meta = []
    unique_idx = []
    for i, dist_meta in enumerate(dist_metas):
        if dist_meta not in unique_dist_meta:
            unique_dist_meta.append(dist_meta)
            unique_idx.append(i)
    return [tensors[i] for i in unique_idx], [dist_metas[i] for i in unique_idx]


def merge_param(tensors: List[Tensor], dist_metas: List[ParamDistMeta]) -> Tensor:
    assert len(tensors) > 0 and len(dist_metas) > 0 and len(tensors) == len(dist_metas)
    # validate parallel info
    validate_parallel_info(dist_metas)
    tensors, dist_metas = deduplicate_params(tensors, dist_metas)
    unflattened_tensors = []
    # group zero params by tp rank
    tensor_dict = defaultdict(list)
    dist_meta_dict = defaultdict(list)
    for t, dist_meta in zip(tensors, dist_metas):
        tensor_dict[dist_meta.tp_rank].append(t)
        dist_meta_dict[dist_meta.tp_rank].append(dist_meta)
    assert len(tensor_dict
              ) == dist_metas[0].tp_world_size, f'Expect {dist_metas[0].tp_world_size} ranks, got {len(tensor_dict)}'
    for tp_rank in tensor_dict.keys():
        unflattened_tensors.append(unflatten_zero_param(tensor_dict[tp_rank], dist_meta_dict[tp_rank]))
    return gather_tp_param(unflattened_tensors, [dist_meta_list[0] for dist_meta_list in dist_meta_dict.values()])


def split_tp_param(tensor: Tensor, redist_meta: ParamRedistMeta) -> List[Tensor]:
    if not redist_meta.used_tp:
        assert redist_meta.tp_world_size == 1, 'Expect tp_world_size == 1, when no tp meta provided.'
        return [tensor]
    total_parts = prod(redist_meta.tp_num_parts)
    assert redist_meta.tp_world_size == total_parts, f'Expect prod(tp_num_parts) == tp_world_size, got {total_parts} and {redist_meta.tp_world_size}.'
    shard_info = sorted(zip(redist_meta.tp_shard_dims, redist_meta.tp_num_parts), key=lambda t: t[0])
    tensors = [tensor]
    for dim, num_parts in shard_info:
        buffer = []
        for t in tensors:
            assert t.size(dim) % num_parts == 0, \
                f'Expect dim{dim} of tensor({tensor.shape}) is divisible by {num_parts}.'
            chunks = [chunk.contiguous() for chunk in t.chunk(num_parts, dim)]
            buffer.extend(chunks)
        tensors = buffer
    assert len(tensors) == redist_meta.tp_world_size
    return tensors


def flatten_zero_param(tensor: Tensor, redist_meta: ParamRedistMeta) -> List[Tensor]:
    if not redist_meta.used_zero:
        return [tensor] * redist_meta.dp_world_size
    tensors: List[Optional[Tensor]] = [
        torch.empty(0, dtype=tensor.dtype, device=tensor.device) for _ in range(redist_meta.zero_start_dp_rank)
    ]
    offsets = redist_meta.zero_offsets + [tensor.numel()]
    for i, offset in enumerate(offsets[:-1]):
        end = offsets[i + 1]
        tensors.append(tensor.view(-1)[offset:end])
    if len(tensors) < redist_meta.dp_world_size:
        tensors.extend([
            torch.empty(0, dtype=tensor.dtype, device=tensor.device)
            for _ in range(redist_meta.dp_world_size - len(tensors))
        ])
    assert len(tensors) == redist_meta.dp_world_size
    return tensors


def unmerge_param(tensor: Tensor, redist_meta: ParamRedistMeta) -> List[List[Tensor]]:
    tensors = split_tp_param(tensor, redist_meta)
    tensors = [flatten_zero_param(t, redist_meta) for t in tensors]
    return tensors

from functools import partial
from typing import Optional

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from utils import assert_dist_model_equal, set_seed

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.sharding_spec import ShardingSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.common import print_rank_0
from colossalai.utils.model.experimental import LazyInitContext, LazyTensor, _MyTensor
from tests.kit.model_zoo import model_zoo


def find_shard_dim(shape: torch.Size) -> Optional[int]:
    for dim, size in enumerate(shape):
        if size % 2 == 0:
            return dim


def make_layout(device_mesh: DeviceMesh, original_tensor: torch.Tensor) -> Layout:
    shard_dim = find_shard_dim(original_tensor.shape)
    # dim_partition_dict = {shard_dim: [0]} if shard_dim is not None else {}
    dim_partition_dict = {}
    target_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict=dim_partition_dict)
    layout = Layout(device_mesh=device_mesh,
                    device_type=torch.device('cuda'),
                    sharding_spec=target_sharding_spec,
                    entire_shape=original_tensor.shape)
    return layout


def _get_current_name(prefix: str, name: str) -> str:
    return f'{prefix}.{name}'.lstrip('.')


def statically_distribute_model(model: nn.Module, device_mesh: DeviceMesh) -> dict:
    # handle shared module
    visited_modules = set()
    layout_dict = {}

    # verbose info
    param_cnt = 0
    param_lazy_cnt = 0
    buf_cnt = 0
    buf_lazy_cnt = 0
    total_numel = 0
    total_lazy_numel = 0

    @torch.no_grad()
    def init_recursively(module: nn.Module, prefix: str = ''):
        nonlocal param_cnt, param_lazy_cnt, buf_cnt, buf_lazy_cnt, total_numel, total_lazy_numel

        # recursively initialize the module
        for name, mod in module.named_children():
            if id(mod) not in visited_modules:
                visited_modules.add(id(mod))
                init_recursively(mod, prefix=_get_current_name(prefix, name))

        # initialize tensors directly attached to the current module
        for name, param in module.named_parameters(recurse=False):
            param_cnt += 1
            total_numel += param.numel()
            if isinstance(param, LazyTensor):
                param_lazy_cnt += 1
                total_lazy_numel += param.numel()

                layout = make_layout(device_mesh, param)
                layout_dict[_get_current_name(prefix, name)] = layout
                # TODO(ver217): apex layers cannot be captured
                setattr(module, name, param.distribute(layout))

        for name, buf in module.named_buffers(recurse=False):
            buf_cnt += 1
            total_numel += buf.numel()
            if isinstance(buf, LazyTensor):
                buf_lazy_cnt += 1
                total_lazy_numel += buf.numel()

                layout = make_layout(device_mesh, buf)
                layout_dict[_get_current_name(prefix, name)] = layout
                setattr(module, name, buf.distribute(layout))

    init_recursively(model)

    print_rank_0(f'Param lazy rate: {param_lazy_cnt}/{param_cnt}')
    print_rank_0(f'Buffer lazy rate: {buf_lazy_cnt}/{buf_cnt}')
    print_rank_0(
        f'Total lazy numel: {total_lazy_numel} ({total_lazy_numel/1024**2:.3f} M), ratio: {total_lazy_numel/total_lazy_numel*100}%'
    )

    return layout_dict


# @parameterize('subset', ['torchvision', 'diffusers', 'timm', 'transformers', 'torchaudio', 'deepfm', 'dlrm'])
@parameterize('subset', ['torchaudio'])
def run_dist_lazy_init(subset, seed: int = 42):
    sub_model_zoo = model_zoo.get_sub_registry(subset)
    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    _MyTensor._pre_op_fn = lambda *args: set_seed(seed)
    LazyTensor._pre_op_fn = lambda *args: set_seed(seed)

    for name, entry in sub_model_zoo.items():
        # TODO(ver217): lazy init does not support weight norm, skip these models
        if name in ('torchaudio_wav2vec2_base', 'torchaudio_hubert_base'):
            continue
        print_rank_0(name)
        model_fn, data_gen_fn, output_transform_fn, model_attr = entry
        torch.cuda.reset_peak_memory_stats()
        ctx = LazyInitContext(tensor_cls=_MyTensor)
        with ctx:
            model = model_fn().cuda()
        print_rank_0(f'Naive init peak cuda mem: {torch.cuda.max_memory_allocated()/1024**2:.3f} MB')
        torch.cuda.reset_peak_memory_stats()
        ctx = LazyInitContext()
        with ctx:
            deferred_model = model_fn().cuda()
        layout_dict = statically_distribute_model(deferred_model, device_mesh)
        print_rank_0(f'Dist lazy init peak cuda mem: {torch.cuda.max_memory_allocated()/1024**2:.3f} MB')
        assert_dist_model_equal(model, deferred_model, layout_dict)


def run_dist(rank, world_size, port) -> None:
    colossalai.launch({}, rank=rank, world_size=world_size, host='localhost', port=port)
    run_dist_lazy_init()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_lazy_init():
    run_func = partial(run_dist, world_size=4, port=free_port())
    mp.spawn(run_func, nprocs=4)


if __name__ == '__main__':
    test_dist_lazy_init()

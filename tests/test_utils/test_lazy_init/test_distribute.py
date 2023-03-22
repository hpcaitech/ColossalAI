from functools import partial
from typing import List, Optional

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from utils import assert_dist_model_equal, set_seed

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor.d_tensor import DTensor
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.layout_converter import to_global
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


def get_percent(numerator: int, denominator: int) -> float:
    if numerator == 0:
        return 0.0
    return numerator / denominator * 100


def make_layout(device_mesh: DeviceMesh, original_tensor: torch.Tensor) -> Layout:
    shard_dim = find_shard_dim(original_tensor.shape)
    dim_partition_dict = {shard_dim: [0]} if shard_dim is not None else {}
    target_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict=dim_partition_dict)
    layout = Layout(device_mesh=device_mesh,
                    device_type=torch.device('cuda'),
                    sharding_spec=target_sharding_spec,
                    entire_shape=original_tensor.shape)
    return layout


def _get_current_name(prefix: str, name: str) -> str:
    return f'{prefix}.{name}'.lstrip('.')


def generate_layout_dict(model: nn.Module, device_mesh: DeviceMesh) -> dict:
    # handle shared module
    visited_modules = set()
    layout_dict = {}

    @torch.no_grad()
    def generate_recursively(module: nn.Module, prefix: str = ''):
        # recursively initialize the module
        for name, mod in module.named_children():
            if id(mod) not in visited_modules:
                visited_modules.add(id(mod))
                generate_recursively(mod, prefix=_get_current_name(prefix, name))

        # initialize tensors directly attached to the current module
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, LazyTensor):
                layout = make_layout(device_mesh, param)
                layout_dict[_get_current_name(prefix, name)] = layout

        for name, buf in module.named_buffers(recurse=False):
            if isinstance(buf, LazyTensor):
                layout = make_layout(device_mesh, buf)
                layout_dict[_get_current_name(prefix, name)] = layout

    generate_recursively(model)

    return layout_dict


def distribute_model(model: nn.Module, layout_dict: dict) -> None:
    # verbose info
    param_cnt = 0
    param_lazy_cnt = 0
    buf_cnt = 0
    buf_lazy_cnt = 0
    total_numel = 0
    non_lazy_numel = 0

    for name, p in model.named_parameters():
        param_cnt += 1
        total_numel += p.numel()
        if getattr(p, '_materialized_data', False) is None:
            # if no _materialized_data attr, the tensor is not lazy
            param_lazy_cnt += 1
        else:
            non_lazy_numel += p.numel()
        if isinstance(p, LazyTensor):
            p.distribute(layout_dict[name])

    for name, buf in model.named_buffers():
        buf_cnt += 1
        total_numel += buf.numel()
        if getattr(buf, "_materialized_data", False) is None:
            # if no _materialized_data attr, the tensor is not lazy
            buf_lazy_cnt += 1
        else:
            non_lazy_numel += buf.numel()
        if isinstance(buf, LazyTensor):
            buf.distribute(layout_dict[name])

    print_rank_0(f'Param lazy rate: {param_lazy_cnt}/{param_cnt}')
    print_rank_0(f'Buffer lazy rate: {buf_lazy_cnt}/{buf_cnt}')
    print_rank_0(
        f'Non lazy numel: {non_lazy_numel} ({non_lazy_numel/1024**2:.3f} M), ratio: {get_percent(non_lazy_numel, total_numel)}%'
    )


@parameterize('subset', ['torchvision', 'diffusers', 'timm', 'transformers', 'torchaudio', 'deepfm', 'dlrm'])
# @parameterize('subset', ['transformers_albert_for_pretraining'])
def run_dist_lazy_init(subset, seed: int = 42):
    sub_model_zoo = model_zoo.get_sub_registry(subset)
    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    _MyTensor._pre_op_fn = lambda *args: set_seed(seed)
    LazyTensor._pre_op_fn = lambda *args: set_seed(seed)

    for name, entry in sub_model_zoo.items():
        # TODO(ver217): lazy init does not support weight norm, skip these models
        if name in ('torchaudio_wav2vec2_base', 'torchaudio_hubert_base', 'timm_coat', 'timm_eca_nfnet',
                    'timm_gmixer_12_224'):
            continue
        print_rank_0(name)
        model_fn, data_gen_fn, output_transform_fn, model_attr = entry
        ctx = LazyInitContext(tensor_cls=_MyTensor)
        with ctx:
            model = model_fn()
        ctx = LazyInitContext()
        with ctx:
            deferred_model = model_fn()
        layout_dict = generate_layout_dict(deferred_model, device_mesh)
        distribute_model(deferred_model, layout_dict)
        assert_dist_model_equal(model, deferred_model, layout_dict)


def run_dist(rank, world_size, port) -> None:
    colossalai.launch({}, rank=rank, world_size=world_size, host='localhost', port=port)
    run_dist_lazy_init()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_lazy_init():
    world_size = 4
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_dist_lazy_init()
    # colossalai.launch_from_torch({})
    # device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    # tensor = torch.rand(3000, device='cuda')
    # layout = make_layout(device_mesh, tensor)
    # print(layout.sharding_spec)
    # d_tensor = DTensor(tensor, layout)
    # print(to_global(d_tensor.local_tensor, layout).shape)

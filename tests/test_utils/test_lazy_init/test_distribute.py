from functools import partial
from typing import Optional

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.sharding_spec import ShardingSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.model.experimental import LazyInitContext, LazyTensor, _MyTensor
from tests.kit.model_zoo import model_zoo


def find_shard_dim(shape: torch.Size) -> Optional[int]:
    for dim, size in enumerate(shape):
        if size % 2 == 0:
            return dim


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


def statically_distribute_model(model: nn.Module, device_mesh: DeviceMesh) -> dict:
    # handle shared module
    visited_modules = set()
    layout_dict = {}

    @torch.no_grad()
    def init_recursively(module: nn.Module, prefix: str = ''):
        # recursively initialize the module
        for name, mod in module.named_children():
            if id(mod) not in visited_modules:
                visited_modules.add(id(mod))
                init_recursively(mod, prefix=_get_current_name(prefix, name))

        # initialize tensors directly attached to the current module
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, LazyTensor):
                layout = make_layout(device_mesh, param)
                layout_dict[_get_current_name(prefix, name)] = layout
                # TODO(ver217): apex layers cannot be captured
                setattr(module, name, param.distribute(layout))

        for name, buf in module.named_buffers(recurse=False):
            if isinstance(buf, LazyTensor):
                layout = make_layout(device_mesh, buf)
                layout_dict[_get_current_name(prefix, name)] = layout
                setattr(module, name, buf.distribute(layout))

    init_recursively(model)

    return layout_dict


@parameterize('subset', ['torchvision', 'diffusers', 'timm', 'transformers', 'torchaudio', 'deepfm', 'dlrm'])
def run_dist_lazy_init(subset):
    sub_model_zoo = model_zoo.get_sub_registry(subset)
    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)

    for name, entry in sub_model_zoo.items():
        # TODO(ver217): lazy init does not support weight norm, skip these models
        if name in ('torchaudio_wav2vec2_base', 'torchaudio_hubert_base'):
            continue
        model_fn, data_gen_fn, output_transform_fn, model_attr = entry
        ctx = LazyInitContext()
        with ctx:
            deferred_model = model_fn()
        statically_distribute_model(deferred_model, device_mesh)


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

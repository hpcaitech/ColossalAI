from typing import Optional

import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.sharding_spec import ShardingSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.common import print_rank_0

try:
    from colossalai.lazy.lazy_init import LazyInitContext, LazyTensor, _MyTensor
except:
    pass
from lazy_init_utils import SUPPORT_LAZY, assert_dist_model_equal, set_seed

from tests.kit.model_zoo import model_zoo


def find_shard_dim(shape: torch.Size) -> Optional[int]:
    for dim, size in enumerate(shape):
        if size % 2 == 0:
            return dim


def make_sharding_spec(original_tensor: torch.Tensor) -> Layout:
    shard_dim = find_shard_dim(original_tensor.shape)
    dim_partition_dict = {shard_dim: [0]} if shard_dim is not None else {}
    target_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict=dim_partition_dict)
    return target_sharding_spec


def _get_current_name(prefix: str, name: str) -> str:
    return f'{prefix}.{name}'.lstrip('.')


def generate_sharding_spec_dict(model: nn.Module) -> dict:
    sharding_spec_dict = {}

    @torch.no_grad()
    def generate_recursively(module: nn.Module, prefix: str = ''):
        # recursively initialize the module
        for name, mod in module.named_children():
            generate_recursively(mod, prefix=_get_current_name(prefix, name))

        # initialize tensors directly attached to the current module
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, LazyTensor):
                sharding_spec = make_sharding_spec(param)
                sharding_spec_dict[_get_current_name(prefix, name)] = sharding_spec

        for name, buf in module.named_buffers(recurse=False):
            if isinstance(buf, LazyTensor):
                sharding_spec = make_sharding_spec(buf)
                sharding_spec_dict[_get_current_name(prefix, name)] = sharding_spec

    generate_recursively(model)

    return sharding_spec_dict


@parameterize('subset', ['torchvision', 'diffusers', 'timm', 'transformers', 'torchaudio', 'deepfm', 'dlrm'])
def run_dist_lazy_init(subset, seed: int = 42):
    sub_model_zoo = model_zoo.get_sub_registry(subset)
    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    _MyTensor._pre_op_fn = lambda *args: set_seed(seed)
    LazyTensor._pre_op_fn = lambda *args: set_seed(seed)

    for name, entry in sub_model_zoo.items():
        # TODO(ver217): lazy init does not support weight norm, skip these models
        if name in ('torchaudio_wav2vec2_base', 'torchaudio_hubert_base') or name.startswith('transformers_llama'):
            continue
        print_rank_0(name)
        model_fn, data_gen_fn, output_transform_fn, _, model_attr = entry
        ctx = LazyInitContext(tensor_cls=_MyTensor)
        with ctx:
            model = model_fn()
        ctx = LazyInitContext()
        with ctx:
            deferred_model = model_fn()
        sharding_spec_dict = generate_sharding_spec_dict(deferred_model)
        ctx.distribute(deferred_model, device_mesh, sharding_spec_dict, verbose=True)
        assert_dist_model_equal(model, deferred_model, device_mesh, sharding_spec_dict)


def run_dist(rank, world_size, port) -> None:
    colossalai.launch({}, rank=rank, world_size=world_size, host='localhost', port=port)
    run_dist_lazy_init()


@pytest.mark.skipif(not SUPPORT_LAZY, reason='torch version should be >= 1.12.0')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_lazy_init():
    spawn(run_dist, 4)


if __name__ == '__main__':
    test_dist_lazy_init()

from functools import partial

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from colossalai.logging import get_dist_logger
from colossalai.testing import parameterize
from colossalai.utils import free_port
from colossalai.context import MOE_CONTEXT
from colossalai.nn.layer import MoeModule
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import (BucketTensorShardStrategy, TensorShardStrategy)

from colossalai.testing import rerun_on_exception
from colossalai.utils import get_current_device
from tests.test_zero_data_parallel.common import CONFIG


class MoeModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(4, 16)
        expert_cls = nn.Linear
        expert_args_dict = dict(in_features=16, out_features=16)
        self.moe = MoeModule(dim_model=16, num_experts=8, use_residual=True, expert_cls=expert_cls, **expert_args_dict)
        self.proj2 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.proj1(x)
        x = self.moe(x)
        x = self.proj2(x)
        return x


@parameterize("init_device_type", ['cpu', 'cuda'])
@parameterize("shard_strategy_class", [TensorShardStrategy, BucketTensorShardStrategy])
def run_moe_zero_init(init_device_type, shard_strategy_class):
    logger = get_dist_logger("test_moe_zero_init")

    if init_device_type == 'cuda':
        init_device = torch.device(f"cuda:{get_current_device()}")
    elif init_device_type == 'cpu':
        init_device = torch.device("cpu")
    else:
        raise NotImplementedError("Unknown device found.")

    model_numel_tensor = torch.zeros(1, dtype=torch.int)
    with ZeroInitContext(target_device=init_device,
                         shard_strategy=shard_strategy_class(),
                         shard_param=True,
                         model_numel_tensor=model_numel_tensor,
                         rm_torch_payload_on_the_fly=False):
        model = MoeModel()

        for name, param in model.named_parameters():
            assert hasattr(param, 'colo_attr')

            # the weights in the gate should be fp32
            if 'gate' in name:
                assert param.colo_attr.sharded_data_tensor.dtype == torch.float32
            else:
                assert param.colo_attr.sharded_data_tensor.dtype == torch.half

            # the parameters in moe experts and its gate should not be sharded
            if ('experts' in name) or ('gate' in name) or ('residual_combine' in name):
                assert not param.colo_attr.sharded_data_tensor.is_sharded
            else:
                assert param.colo_attr.sharded_data_tensor.is_sharded

            # the parameters in moe experts is not replicated
            if 'experts' in name:
                assert not param.is_replicated
            else:
                assert param.is_replicated

            assert param.colo_attr.sharded_data_tensor.payload.device.type == init_device.type, \
                f'{param.colo_attr.sharded_data_tensor.payload.device.type} vs. {init_device.type}'


def _run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    run_moe_zero_init()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_moe_zero_init(world_size):
    run_func = partial(_run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_moe_zero_init(world_size=2)

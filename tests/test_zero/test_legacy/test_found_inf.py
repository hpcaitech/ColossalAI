import pytest
import torch
from common import CONFIG
from test_sharded_optim_v2 import _run_step

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import BucketTensorShardStrategy
from colossalai.zero.legacy.sharded_model import ShardedModelV2
from colossalai.zero.legacy.sharded_optim import ShardedOptimizerV2
from colossalai.zero.low_level._utils import has_inf_or_nan
from tests.components_to_test.registry import non_distributed_component_funcs


@parameterize("cpu_offload", [True, False])
@parameterize("shard_strategy_class", [BucketTensorShardStrategy])
@parameterize("gpu_margin_mem_ratio", [0.0, 0.7])
def _run_test_found_inf(cpu_offload, shard_strategy_class, gpu_margin_mem_ratio):
    test_models = ['repeated_computed_layers']
    shard_strategy = shard_strategy_class()

    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, optimizer_class, criterion = get_components_func()

        with ZeroInitContext(target_device=torch.device(f'cpu:0') if cpu_offload else get_current_device(),
                             shard_strategy=shard_strategy,
                             shard_param=True):
            zero_model = model_builder(checkpoint=True)
        zero_model = ShardedModelV2(
            zero_model,
            shard_strategy,
            tensor_placement_policy='cpu' if cpu_offload else 'cuda',
            reuse_fp16_shard=True,
        )

        sharded_optim = HybridAdam(zero_model.parameters(), lr=1e-3)
        sharded_optim = ShardedOptimizerV2(zero_model, sharded_optim, gpu_margin_mem_ratio=gpu_margin_mem_ratio)

        for i, (data, label) in enumerate(train_dataloader):
            if i > 1:
                break
            assert zero_model.overflow_counter == 0
            data, label = data.cuda(), label.cuda()
            _run_step(zero_model, sharded_optim, data, label, criterion, False)
            for param in zero_model.parameters():
                assert not has_inf_or_nan(param.colo_attr.data_payload)


def _run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_test_found_inf()


# use_cpuadam = True can be used with cpu_offload = False
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_found_inf(world_size):
    spawn(_run_dist, world_size)


if __name__ == '__main__':
    test_found_inf(world_size=2)

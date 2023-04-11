import pytest
import torch

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ProcessGroup
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero.legacy.init_ctx import ZeroInitContext
from colossalai.zero.legacy.shard_utils import TensorShardStrategy
from colossalai.zero.legacy.sharded_model import ShardedModelV2
from colossalai.zero.legacy.sharded_optim import ShardedOptimizerV2
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import set_seed


def init_zero(model_builder, placement_policy):
    device = get_current_device() if placement_policy == 'cuda' else torch.device('cpu')
    shard_strategy = TensorShardStrategy()
    with ZeroInitContext(target_device=device, shard_strategy=shard_strategy, shard_param=True):
        model = model_builder()
    model = ShardedModelV2(
        model,
        shard_strategy,
        tensor_placement_policy=placement_policy,
        reuse_fp16_shard=True,
    )
    optim = HybridAdam(model.parameters(), lr=1e-3)
    optim = ShardedOptimizerV2(model, optim, initial_scale=32)
    return model, optim


def run_step(model, optim, criterion, data, label):
    optim.zero_grad()
    logits = model(data)
    loss = criterion(logits, label)
    optim.backward(loss)
    optim.step()


def check_state_dict_eq(state_dict, other):
    for p, state in state_dict['state'].items():
        other_state = other['state'][p]
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                assert torch.allclose(v, other_state[k], atol=1e-3), f'{v} vs {other_state[k]}'
            else:
                assert v == other_state[k]


@parameterize('placement_policy', ['cuda', 'cpu'])
def run_nested_model(placement_policy):
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    set_seed(42)
    model, optim = init_zero(model_builder, placement_policy)
    set_seed(42)
    model_copy, optim_copy = init_zero(model_builder, placement_policy)

    model.train()
    model_copy.train()
    pg = ProcessGroup()
    set_seed(pg.dp_local_rank())
    data_iter = iter(train_dataloader)

    data, label = map(lambda x: x.cuda(), next(data_iter))
    run_step(model, optim, criterion, data, label)
    optim_copy.load_state_dict(optim.state_dict())
    check_state_dict_eq(optim.state_dict(), optim_copy.state_dict())

    data, label = map(lambda x: x.cuda(), next(data_iter))
    run_step(model_copy, optim_copy, criterion, data, label)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_nested_model()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_sharded_optim_state_dist(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_sharded_optim_state_dist(2)

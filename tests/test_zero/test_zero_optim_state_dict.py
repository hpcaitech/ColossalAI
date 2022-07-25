import pytest
import colossalai
import torch
from colossalai.context.parallel_mode import ParallelMode
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.core import global_context as gpc
from functools import partial
from tests.test_tensor.common_utils import set_seed
from tests.components_to_test.registry import non_distributed_component_funcs
from colossalai.nn.parallel.data_parallel import ZeroDDP
from colossalai.gemini import ChunkManager, GeminiManager
from colossalai.testing import parameterize
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer
from colossalai.tensor import ProcessGroup


def init_zero(model, use_chunk, use_zero, placement_policy):
    pg = ProcessGroup()
    chunk_size = ChunkManager.search_chunk_size(model, 8192, 8) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size,
                                 pg,
                                 enable_distributed_storage=use_zero,
                                 init_device=GeminiManager.get_default_device(placement_policy))
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    return ZeroDDP(model, gemini_manager)


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


@parameterize('use_chunk', [False, True])
@parameterize('use_zero', [False, True])
@parameterize('placement_policy', ['cuda', 'cpu'])
def run_nested_model(use_chunk, use_zero, placement_policy):
    get_components_func = non_distributed_component_funcs.get_callable('nested_model')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    set_seed(42)
    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    set_seed(42)
    with ColoInitContext(device=get_current_device()):
        model_copy = model_builder()
    model = init_zero(model, use_chunk, use_zero, placement_policy)
    model_copy = init_zero(model_copy, use_chunk, use_zero, placement_policy)

    optim = HybridAdam(model.parameters(), lr=1e-3)
    optim = ZeroOptimizer(optim, model, initial_scale=32)
    optim_copy = HybridAdam(model_copy.parameters(), lr=1e-3)
    optim_copy = ZeroOptimizer(optim_copy, model_copy, initial_scale=32)

    model.train()
    model_copy.train()
    set_seed(gpc.get_local_rank(ParallelMode.DATA))
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
def test_zero_optim_state_dist(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_optim_state_dist(2)

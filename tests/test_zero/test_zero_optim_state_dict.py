import pytest
import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.gemini import ChunkManager
from functools import partial
from tests.components_to_test.registry import non_distributed_component_funcs
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer
from colossalai.testing import parameterize
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.tensor import ProcessGroup


def check_state(s1, s2):
    for v1, v2 in zip(s1.values(), s2.values()):
        if isinstance(v1, torch.Tensor):
            v1 = v1.to(v2.device)
            assert torch.equal(v1, v2), f'{torch.sum((v1-v2).abs())}'
        else:
            assert v1 == v2


def check_load_state_dict(optim, torch_optim):
    for group, torch_group in zip(optim.optim.param_groups, torch_optim.param_groups):
        for p, torch_p in zip(group['params'], torch_group['params']):
            state = optim.optim.state[p]
            torch_state = torch_optim.state[torch_p]
            if p.storage().size() == 0:
                assert len(state) == 0
            check_state(state, torch_state)


def check_state_dict(state_dict, torch_state_dict):
    for (k1, s1), (k2, s2) in zip(state_dict['state'].items(), torch_state_dict['state'].items()):
        assert k1 == k2
        check_state(s1, s2)


@parameterize('use_chunk', [False, True])
@parameterize('use_zero', [False, True])
@parameterize('placement_policy', ['cuda', 'cpu', 'auto'])
def run_zero_optim_state_dict(use_chunk, use_zero, placement_policy):
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = model.cuda()
    torch_model = model_builder().cuda()

    pg = ProcessGroup()

    chunk_size = ChunkManager.search_chunk_size(model, 8192, 8) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size,
                                 pg,
                                 enable_distributed_storage=use_zero,
                                 init_device=GeminiManager.get_default_device(placement_policy))
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager)
    optim = HybridAdam(model.parameters(), lr=1e-3)
    optim = ZeroOptimizer(optim, model, initial_scale=1)

    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)

    for p in torch_model.parameters():
        p.grad = torch.rand_like(p)

    torch_optim.step()
    torch_state_dict = torch_optim.state_dict()
    optim.load_state_dict(torch_state_dict)
    check_load_state_dict(optim, torch_optim)

    state_dict = optim.state_dict()
    if pg.rank() == 0:
        check_state_dict(state_dict, torch_state_dict)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_zero_optim_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_zero_optim_state_dict(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_optim_state_dict(2)

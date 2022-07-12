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
from colossalai.nn.parallel import ZeroDDP, ColoDDP
from colossalai.gemini.gemini_mgr import GeminiManager
from typing import Callable
from collections import OrderedDict
from colossalai.tensor import ProcessGroup, ColoParameter


def check_state_dict_equal(state_dict: OrderedDict, other_state_dict: OrderedDict):
    for (k1, t1), (k2, t2) in zip(state_dict.items(), other_state_dict.items()):
        assert k1 == k2
        assert torch.allclose(t1, t2, atol=1e-3, rtol=1e-3)


def init_ddp(module: torch.nn.Module) -> ColoDDP:
    pg = ProcessGroup()
    return ColoDDP(module, process_group=pg)


def init_ddpv2(module: torch.nn.Module, use_chunk: bool = False, use_zero: bool = False) -> ZeroDDP:
    chunk_size = ChunkManager.search_chunk_size(module, 64, 4) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size, enable_distributed_storage=use_zero)
    gemini_manager = GeminiManager('cuda', chunk_manager)
    pg = ProcessGroup()
    return ZeroDDP(module, gemini_manager, process_group=pg)


def run_state_dict(ddp_init_func: Callable[[torch.nn.Module], ColoDDP]):
    get_components_func = non_distributed_component_funcs.get_callable('nested_model')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    torch_model = model_builder().cuda()
    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = ddp_init_func(model)
    torch_state_dict = torch_model.state_dict()
    for param in model.parameters():
        if isinstance(param, ColoParameter):
            assert param.get_process_group() is not None
    model.load_state_dict(torch_state_dict)

    for param in model.parameters():
        if isinstance(param, ColoParameter):
            assert param.get_process_group() is not None

    state_dict = model.state_dict()
    check_state_dict_equal(torch_state_dict, state_dict)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_state_dict(init_ddp)
    run_state_dict(partial(init_ddpv2, use_chunk=False, use_zero=False))
    run_state_dict(partial(init_ddpv2, use_chunk=False, use_zero=True))
    run_state_dict(partial(init_ddpv2, use_chunk=True, use_zero=False))
    run_state_dict(partial(init_ddpv2, use_chunk=True, use_zero=True))


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_state_dict(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_state_dict(2)

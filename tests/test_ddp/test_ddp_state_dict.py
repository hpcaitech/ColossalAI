import copy

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
from collections import OrderedDict
from colossalai.tensor import ProcessGroup, ColoParameter
from colossalai.testing import parameterize


def check_state_dict_equal(state_dict: OrderedDict, other_state_dict: OrderedDict):
    for (k1, t1), (k2, t2) in zip(state_dict.items(), other_state_dict.items()):
        assert k1 == k2

        if t1.device != t2.device:
            temp_t2 = t2.to(t1.device)
        else:
            temp_t2 = t2

        assert torch.equal(t1, temp_t2), "\t{}\n\t{}".format(t1, temp_t2)


def check_model_equal(model_a, model_b, allow_empty: bool = False, same_dtype: bool = True):
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert na == nb

        if not allow_empty:
            assert pa.storage().size() > 0
            assert pb.storage().size() > 0
        else:
            if pa.storage().size() == 0 or pb.storage().size() == 0:
                continue

        if same_dtype:
            assert pa.dtype == pb.dtype
            temp_pb = pb
        else:
            temp_pb = pb.to(pa.dtype)

        assert torch.equal(pa, temp_pb), "Parameter '{}' is not equal.\n {} {}".format(na, pa, pb)


def init_ddp(module: torch.nn.Module) -> ColoDDP:
    pg = ProcessGroup()
    return ColoDDP(module, process_group=pg)


def init_ddpv2(module: torch.nn.Module,
               use_chunk: bool = False,
               use_zero: bool = False,
               placement_policy: str = 'cuda') -> ZeroDDP:
    pg = ProcessGroup()
    chunk_size = ChunkManager.search_chunk_size(module, 64, 4) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=use_zero)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    return ZeroDDP(module, gemini_manager)


def run_ddp_state_dict():
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    torch_model = model_builder().cuda()
    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = init_ddp(model)
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


@parameterize('use_chunk', [False, True])
@parameterize('placement_policy', ['cuda', 'cpu'])
@parameterize('use_zero', [False, True])
@parameterize('only_rank_0', [False, True])
def run_zero_state_dict(use_chunk, placement_policy, use_zero, only_rank_0):
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    torch_model = model_builder().cuda()
    org_torch_model = copy.deepcopy(torch_model)
    torch_state_dict = torch_model.state_dict()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = init_ddpv2(model, use_chunk, use_zero, placement_policy)

    for param in model.parameters():
        if isinstance(param, ColoParameter):
            assert param.get_process_group() is not None

    model.load_state_dict(torch_state_dict, strict=False)
    check_model_equal(model, torch_model, allow_empty=True, same_dtype=False)

    for param in model.parameters():
        if isinstance(param, ColoParameter):
            assert param.get_process_group() is not None

    pg = ProcessGroup()
    state_dict = model.state_dict(only_rank_0=only_rank_0)
    if not only_rank_0 or pg.dp_local_rank() == 0:
        torch_model.load_state_dict(state_dict, strict=False)
        check_model_equal(torch_model, org_torch_model, allow_empty=False, same_dtype=True)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_ddp_state_dict()
    run_zero_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_state_dict(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_state_dict(2)

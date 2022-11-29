from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import colossalai
from colossalai.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.nn.parallel import ZeroDDP
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import debug_print, set_seed


@parameterize('placement_policy', ['cuda', 'cpu', 'auto'])
@parameterize('keep_gathered', [True, False])
@parameterize('model_name', ['gpt2', 'bert'])
def exam_state_dict(placement_policy, keep_gathered, model_name: str):
    set_seed(431)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()

    torch_model = model_builder()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = keep_gathered
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)
    model.train()

    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        if key == 'model.lm_head.weight':
            continue
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        assert torch.equal(value, temp_zero_value), "parameter '{}' has problem.".format(key)


@parameterize('placement_policy', ['cuda', 'cpu', 'auto'])
@parameterize('keep_gathered', [True, False])
@parameterize('model_name', ['gpt2', 'bert'])
def exam_load_state_dict(placement_policy, keep_gathered, model_name: str):
    set_seed(431)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()

    set_seed(451)
    torch_model = model_builder()    # get a different model

    world_size = torch.distributed.get_world_size()
    config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = keep_gathered

    if placement_policy != 'cuda':
        init_device = torch.device('cpu')
    else:
        init_device = None
    chunk_manager = ChunkManager(config_dict, init_device=init_device)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)

    torch_dict = torch_model.state_dict()
    model.load_state_dict(torch_dict, strict=False)
    zero_dict = model.state_dict(only_rank_0=False)

    for key, value in torch_dict.items():
        if key == 'model.lm_head.weight':
            continue
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        assert torch.equal(value, temp_zero_value), "parameter '{}' has problem.".format(key)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_state_dict()
    exam_load_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_zero_ddp(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_ddp(1)

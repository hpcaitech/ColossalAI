from typing import Callable

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing import assert_close

import colossalai
from colossalai.amp import convert_to_apex_amp
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext, ZeroDDP, ZeroOptimizer, post_process_colo_init_ctx, zero_model_wrapper
from colossalai.zero.gemini.chunk import ChunkManager, init_chunk_manager, search_chunk_configuration
from colossalai.zero.gemini.gemini_mgr import GeminiManager
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import debug_print, set_seed


def check_param(model: ZeroDDP, torch_model: torch.nn.Module):
    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        # key is 'module.model.PARAMETER', so we truncate it
        key = key[7:]
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        # debug_print([0], "max range: ", key, torch.max(torch.abs(value - temp_zero_value)))
        assert_close(value, temp_zero_value, rtol=1e-3, atol=4e-3)


def multi_chunk_init(model: torch.nn.Module, placement_policy: str):
    world_size = dist.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = False
    if placement_policy != 'cuda':
        init_device = torch.device('cpu')
    else:
        init_device = None
    chunk_manager = ChunkManager(config_dict, init_device=init_device)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)
    return model


def single_chunk_init(model: torch.nn.Module, placement_policy: str):
    gemini_config = dict(
        device=get_current_device(),
        placement_policy=placement_policy,
        pin_memory=True,
    )
    model = zero_model_wrapper(model=model, zero_stage=3, gemini_config=gemini_config)
    return model


@parameterize('placement_policy', ['cuda', 'cpu', 'auto', 'const'])
@parameterize('model_name', ['gpt2'])
@parameterize('model_init_func', [single_chunk_init, multi_chunk_init])
def exam_inference(placement_policy: str, model_name: str, model_init_func: Callable):
    set_seed(19360226)
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    torch_model = model_builder().cuda()
    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=128)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[dist.get_rank()])

    init_dev = get_current_device()
    with ColoInitContext(device=init_dev):
        model = model_builder()

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        p.data.copy_(torch_p.data)

    model = model_init_func(model, placement_policy)
    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = ZeroOptimizer(optimizer, model, initial_scale=128)

    model.eval()
    torch_model.eval()

    set_seed(dist.get_rank() * 3 + 128)
    train_dataloader = iter(train_dataloader)

    def train_iter():
        input_ids, label = next(train_dataloader)
        input_ids, label = input_ids.cuda(), label.cuda()
        zero_optim.zero_grad()
        torch_optim.zero_grad()
        torch_loss = run_fwd_bwd(torch_model, input_ids, label, criterion, torch_optim)
        loss = run_fwd_bwd(model, input_ids, label, criterion, zero_optim)
        assert_close(torch_loss, loss)
        zero_optim.step()
        torch_optim.step()
        check_param(model, torch_model)

    def inference_iter():
        input_ids, label = next(train_dataloader)
        input_ids, label = input_ids.cuda(), label.cuda()
        with torch.no_grad():
            torch_output = torch_model(input_ids)
            torch_loss = criterion(torch_output.float(), label)
            zero_output = model(input_ids)
            zero_loss = criterion(zero_output.float(), label)
        assert_close(torch_loss, zero_loss)

    train_iter()
    inference_iter()
    train_iter()


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_inference()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_inference(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_inference(1)

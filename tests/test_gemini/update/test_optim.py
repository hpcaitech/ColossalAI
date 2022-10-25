from functools import partial
from time import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.amp import convert_to_apex_amp
from colossalai.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import ZeroDDP
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.zero import ZeroOptimizer
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import debug_print, set_seed, tensor_equal, tensor_shard_equal


def check_param(model: ZeroDDP, torch_model: torch.nn.Module):
    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        # key is 'module.model.PARAMETER', so we truncate it
        key = key[7:]
        if key == 'model.lm_head.weight':
            continue
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        # debug_print([0], "max range: ", key, torch.max(torch.abs(value - temp_zero_value)))
        assert torch.allclose(value, temp_zero_value, rtol=1e-3, atol=1e-2), "parameter '{}' has problem.".format(key)


def run_fwd_bwd(model, criterion, optimizer, input_ids, attn_mask):
    optimizer.zero_grad()
    logits = model(input_ids, attn_mask)
    logits = logits.float()
    loss = criterion(logits, input_ids)
    optimizer.backward(loss)
    return logits


@parameterize('placement_policy', ['cuda', 'cpu', 'auto', 'const'])
def exam_gpt_fwd_bwd(placement_policy):
    set_seed(42)
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()

    torch_model = model_builder().cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)

    world_size = torch.distributed.get_world_size()
    config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = False
    if placement_policy != 'cuda':
        init_device = torch.device('cpu')
    else:
        init_device = None
    chunk_manager = ChunkManager(config_dict, init_device=init_device)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)

    optimizer = HybridAdam(model.parameters(), lr=1e-3)
    zero_optim = ZeroOptimizer(optimizer, model, initial_scale=2)

    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=1)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[dist.get_rank()])

    model.eval()
    torch_model.eval()

    set_seed(dist.get_rank() * 3 + 128)
    for i, (input_ids, attn_mask) in enumerate(train_dataloader):
        if i > 2:
            break

        zero_logits = run_fwd_bwd(model, criterion, zero_optim, input_ids, attn_mask)
        torch_logits = run_fwd_bwd(torch_model, criterion, torch_optim, input_ids, attn_mask)
        assert torch.allclose(zero_logits, torch_logits, rtol=1e-3, atol=1e-2)
        # debug_print([0], zero_logits, torch_logits)

        zero_optim.step()
        torch_optim.step()

        check_param(model, torch_model)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_gpt_fwd_bwd()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gpt(1)

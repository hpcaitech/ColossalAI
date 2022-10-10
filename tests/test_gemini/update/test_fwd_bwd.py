import pytest
import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext

from functools import partial
from tests.test_tensor.common_utils import tensor_equal, set_seed, tensor_shard_equal
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP
from colossalai.nn.parallel import ZeroDDP
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer
from colossalai.testing import parameterize
from colossalai.amp import convert_to_apex_amp
from colossalai.gemini.gemini_mgr import GeminiManager
from colossalai.tensor import ColoTensorSpec, ShardSpec, ComputePattern, ComputeSpec, ProcessGroup, ColoTensor
from tests.test_tensor.common_utils import debug_print

from time import time
from colossalai.gemini.chunk import search_chunk_configuration, ChunkManager


def check_grad(model: ZeroDDP, torch_model: torch.nn.Module):
    chunk_manager = model.chunk_manager
    param_list = [p for p in model.parameters()]
    chunk_list = chunk_manager.get_chunks(param_list)
    for chunk in chunk_list:
        chunk_manager.access_chunk(chunk)

    for (p0, p1) in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(p0, p1.grad, atol=1e-3, rtol=1e-5), "{}".format(torch.max(torch.abs(p0 - p1.grad)).item())


def run_fwd_bwd(model, criterion, optimizer, input_ids, attn_mask):
    optimizer.zero_grad()
    logits = model(input_ids, attn_mask)
    logits = logits.float()
    loss = criterion(logits, input_ids)
    optimizer.backward(loss)
    return logits


@parameterize('placement_policy', ['cuda', 'cpu', 'auto'])
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
    config_dict = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = False
    chunk_manager = ChunkManager(config_dict)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)

    pg = ProcessGroup()
    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=1)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[pg.rank()], process_group=pg.dp_process_group())

    model.eval()
    torch_model.eval()

    set_seed(pg.dp_local_rank())
    for i, (input_ids, attn_mask) in enumerate(train_dataloader):
        if i > 0:
            break

        logits = model(input_ids, attn_mask)
        logits = logits.float()
        loss = criterion(logits, input_ids)
        model.backward(loss)

        torch_logits = run_fwd_bwd(torch_model, criterion, torch_optim, input_ids, attn_mask)
        assert torch.allclose(logits, torch_logits, rtol=0), "{} {} {}".format(
            torch.max(torch.abs(logits - torch_logits)).item(), logits, torch_logits)

        check_grad(model, torch_model)


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

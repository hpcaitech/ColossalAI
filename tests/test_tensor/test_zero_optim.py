import pytest
import colossalai
import torch
import torch.multiprocessing as mp
from colossalai.context.parallel_mode import ParallelMode
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ChunkManager
from colossalai.core import global_context as gpc
from functools import partial
from _utils import tensor_equal, set_seed
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP
from colossalai.nn.parallel import ColoDDPV2
from colossalai.nn.optimizer import HybridAdam
from colossalai.zero import ZeroOptimizer
from colossalai.testing import parameterize
from colossalai.amp import convert_to_apex_amp
from colossalai.gemini.gemini_mgr import GeminiManager


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        if p.storage().size() > 0:
            assert p.dtype == torch.half
            assert tensor_equal(torch_p.to(dtype=p.dtype, device=p.device), p), f'{torch_p} vs {p}'


def check_grad_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        if p.grad is not None:
            assert tensor_equal(torch_p.grad.to(dtype=p.grad.dtype, device=p.grad.device), p.grad)


def run_fwd_bwd(model, criterion, optimizer, input_ids, attn_mask):
    optimizer.zero_grad()
    logits = model(input_ids, attn_mask)
    logits = logits.float()
    loss = criterion(logits, input_ids)
    optimizer.backward(loss)
    return logits


@parameterize('use_chunk', [False, True])
@parameterize('use_zero', [False, True])
@parameterize('placement_policy', ['cuda', 'cpu'])
def run_gpt(use_chunk, use_zero, placement_policy):
    set_seed(42)
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = model.cuda().half()
    torch_model = model_builder().cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p)

    chunk_size = ChunkManager.search_chunk_size(model, 8192, 8) if use_chunk else None
    chunk_manager = ChunkManager(chunk_size,
                                 enable_distributed_storage=use_zero,
                                 init_device=GeminiManager.get_default_device(placement_policy))
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ColoDDPV2(model, gemini_manager)
    optim = HybridAdam(model.parameters(), lr=1e-3)
    optim = ZeroOptimizer(optim, model, initial_scale=32)

    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=32)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[gpc.get_global_rank()], process_group=gpc.get_group(ParallelMode.DATA))

    print(chunk_manager)
    check_param_equal(model, torch_model)
    model.train()
    torch_model.train()
    set_seed(gpc.get_local_rank(ParallelMode.DATA))
    for i, (input_ids, attn_mask) in enumerate(train_dataloader):
        if i > 2:
            break
        logits = run_fwd_bwd(model, criterion, optim, input_ids, attn_mask)
        torch_logits = run_fwd_bwd(torch_model, criterion, torch_optim, input_ids, attn_mask)
        assert tensor_equal(logits, torch_logits)
        check_grad_equal(model, torch_model)
        optim.step()
        torch_optim.step()
        check_param_equal(model, torch_model)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gpt(4)

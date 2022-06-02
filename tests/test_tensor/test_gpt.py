import pytest
import colossalai
from colossalai.context.parallel_mode import ParallelMode
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, DistSpecManager, distspec
from colossalai.core import global_context as gpc
from functools import partial
from _utils import tensor_equal, tensor_shard_equal, set_seed
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP
from colossalai.nn.parallel import ColoDDP


def init_1d_row_spec(model):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n and 'ln' not in n:
                p.set_spec(spec)


def init_1d_col_spec(model):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'ln' not in n and ('weight' in n or 'bias' in n):
                p.set_spec(spec)


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert tensor_shard_equal(torch_p, p)


def check_grad_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert tensor_shard_equal(torch_p.grad, p.grad)


def run_gpt(init_spec_func, use_ddp):
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = model.cuda()
    torch_model = model_builder().cuda()
    if use_ddp:
        model = ColoDDP(model)
        torch_model = DDP(torch_model,
                          device_ids=[gpc.get_global_rank()],
                          process_group=gpc.get_group(ParallelMode.DATA))
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p)
    init_spec_func(model)
    check_param_equal(model, torch_model)
    model.train()
    torch_model.train()
    set_seed(gpc.get_local_rank(ParallelMode.DATA))
    for i, (input_ids, attn_mask) in enumerate(train_dataloader):
        logits = model(input_ids, attn_mask)
        torch_logits = torch_model(input_ids, attn_mask)
        assert tensor_equal(torch_logits, logits)
        loss = criterion(logits, input_ids)
        torch_loss = criterion(torch_logits, input_ids)
        if use_ddp:
            model.backward(loss)
        else:
            loss.backward()
        torch_loss.backward()
        check_grad_equal(model, torch_model)
        if i > 0:
            break


def run_dist(rank, world_size, port, use_ddp):
    if use_ddp and world_size == 1:
        return
    tp_world_size = world_size // 2 if use_ddp else world_size
    config = dict(parallel=dict(tensor=dict(mode="1d", size=tp_world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt(init_1d_row_spec, use_ddp)
    run_gpt(init_1d_col_spec, use_ddp)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_ddp', [False, True])
@rerun_if_address_is_in_use()
def test_gpt(world_size, use_ddp):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), use_ddp=use_ddp)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gpt(4)

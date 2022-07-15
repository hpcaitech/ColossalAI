import pytest

from functools import partial
from _utils import tensor_equal, tensor_shard_equal, set_seed

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import colossalai
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ShardSpec, ComputePattern, ComputeSpec, ProcessGroup, ColoTensor, ColoTensorSpec
from colossalai.nn.parallel.data_parallel import ColoDDP
from tests.components_to_test.registry import non_distributed_component_funcs


def init_1d_row_spec(model, pg: ProcessGroup):
    tensor_spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    for n, p in model.named_parameters():
        p.set_process_group(pg)
        if 'weight' in n and 'ln' not in n:
            p.set_tensor_spec(*tensor_spec)


def init_1d_col_spec(model, pg: ProcessGroup):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))

    for n, p in model.named_parameters():
        p.set_process_group(pg)
        if 'ln' not in n and ('weight' in n or 'bias' in n):
            p.set_tensor_spec(*spec)


def check_param_equal(model, torch_model, pg: ProcessGroup):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert pg.tp_local_rank() is not None, f"{pg.rank()} {pg.tp_world_size()} {pg._tp_degree} {pg.tp_local_rank()}1"
        assert pg.tp_world_size() is not None
        assert tensor_shard_equal(torch_p, p, pg.tp_local_rank(), pg.tp_world_size())


def check_grad_equal(model, torch_model, pg: ProcessGroup):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert tensor_shard_equal(torch_p.grad, p.grad, pg.tp_local_rank(), pg.tp_world_size())


def run_gpt(init_spec_func, use_ddp):
    world_size = torch.distributed.get_world_size()

    # build a PG with TP and DP hybrid
    pg = ProcessGroup(dp_degree=(2 if (use_ddp and world_size >= 2) else 1))

    # set seed make processes of the same tp group use the same seed
    # set_seed(pg.tp_local_rank())

    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    # make sure torch_model and model has the same parameter values
    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = model.cuda()
    torch_model = model_builder().cuda()

    if use_ddp:
        torch_model = DDP(torch_model, device_ids=[pg.rank()], process_group=pg.dp_process_group())
        model = ColoDDP(model, process_group=pg)

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p)

    init_spec_func(model, pg)

    check_param_equal(model, torch_model, pg)

    # close the dropout in eval mode
    model.eval()
    torch_model.eval()
    set_seed(pg.dp_local_rank())
    torch.distributed.barrier()
    for i, (input_ids, attn_mask) in enumerate(train_dataloader):
        colo_input = ColoTensor.from_torch_tensor(input_ids, ColoTensorSpec(pg))
        logits = model(colo_input, attn_mask)
        torch_logits = torch_model(input_ids, attn_mask)
        assert tensor_equal(torch_logits, logits), f"{torch_logits - logits}"
        loss = criterion(logits, input_ids)
        torch_loss = criterion(torch_logits, input_ids)
        if use_ddp:
            model.backward(loss)
        else:
            loss.backward()
        torch_loss.backward()
        check_grad_equal(model, torch_model, pg)
        if i > 0:
            break
    set_seed(313)


def run_dist(rank, world_size, port, use_ddp):
    if use_ddp and world_size == 1:
        return
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
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
    test_gpt(4, use_ddp=True)

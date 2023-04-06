import pytest
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.tensor import ColoTensor, ColoTensorSpec, ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import (
    debug_print,
    set_seed,
    split_param_col_tp1d,
    split_param_row_tp1d,
    tensor_equal,
    tensor_shard_equal,
)


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


def init_megatron_spec(model, pg: ProcessGroup):
    for mn, module in model.named_modules():
        # debug_print([0], mn)
        for pn, param in module.named_parameters(recurse=False):
            # debug_print([0], '\t', pn, param.compute_spec, param.shape)
            param.set_process_group(pg)

            if 'mlp.c_fc' in mn:
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)
                    param.compute_spec.set_output_replicate(False)
                else:
                    raise RuntimeError
            elif 'mlp.c_proj' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)
                else:
                    assert 'bias' in pn
            elif 'wte' in mn or 'wpe' in mn:
                assert 'weight' in pn
                split_param_col_tp1d(param, pg)
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)
            # debug_print([0], '\t', param.compute_spec, param.shape)


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
    for i, (input_ids, label) in enumerate(train_dataloader):
        colo_input = ColoTensor.from_torch_tensor(input_ids, ColoTensorSpec(pg))
        logits = model(colo_input)
        torch_logits = torch_model(input_ids)
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
    # Comments below tests for speed concern
    # run_gpt(init_1d_row_spec, use_ddp)
    # run_gpt(init_1d_col_spec, use_ddp)
    run_gpt(init_megatron_spec, use_ddp)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_ddp', [False, True])
@rerun_if_address_is_in_use()
def test_gpt(world_size, use_ddp):
    spawn(run_dist, world_size, use_ddp=use_ddp)


if __name__ == '__main__':
    test_gpt(4, use_ddp=False)

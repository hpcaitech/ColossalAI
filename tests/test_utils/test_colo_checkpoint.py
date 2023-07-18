import os
import shutil
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR

import colossalai
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.tensor import ColoTensor, ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.checkpoint import load_checkpoint, save_checkpoint
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs


def init_1d_row_linear(weight: ColoTensor, pg: ProcessGroup):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    weight.set_process_group(pg)
    weight.set_tensor_spec(*spec)


def init_1d_col_linear(weight, pg):
    spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    weight.set_process_group(pg)
    weight.set_tensor_spec(*spec)


def init_1d_row_embedding(weight, pg):
    spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    weight.set_process_group(pg)
    weight.set_tensor_spec(*spec)


def init_1d_col_embedding(weight, pg):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    weight.set_process_group(pg)
    weight.set_tensor_spec(*spec)


def init_1d_row_for_linear_weight_spec(model, pg: ProcessGroup):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    for name, p in model.named_parameters():
        if not isinstance(p, ColoTensor):
            continue
        if 'embed' in name and 'weight' in name:
            init_1d_col_embedding(p, pg)
        if 'proj1' in name and ('weight' in name or 'bias' in name):
            init_1d_col_linear(p, pg)
        if 'proj2' in name and 'weight' in name:
            init_1d_row_linear(p, pg)
        if 'classifier' in name and ('weight' in name or 'bias' in name):
            init_1d_col_linear(p, pg)


def check_param_equal(model, torch_model):
    for (n, p), (tn, tp) in zip(model.named_parameters(), torch_model.named_parameters()):
        assert torch.all(p.data == tp.data), "{} went wrong.\n {} vs {}\n{}".format(n, p, tp, p.shape)


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def compare_optims(optim1, optim2):
    state1 = optim1.state_dict()['state']
    state2 = optim2.state_dict()['state']
    for k, p1 in state1.items():
        if k not in state2:
            continue
        p2 = state2[k]
        for n, t1 in p1.items():
            if n not in p2:
                continue
            t2 = p2[n]
            if isinstance(t1, ColoTensor):
                assert isinstance(t2, ColoTensor)
                assert torch.allclose(t1, t2, rtol=0, atol=0)


def _run_checkpoint(model_name, init_spec_func, use_ddp, use_mp_reload, test_scheduler, pg):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    if use_mp_reload:
        if 'bert' == model_name:
            for name, p in model.named_parameters():
                if not isinstance(p, ColoTensor):
                    continue
                # num_class = type_vocab_size = 2 | (8, 2)
                if 'classifier' in name and 'weight' in name:
                    init_1d_row_linear(p, pg)
                # num_class = vocab_size = 30524 | (30524, 8)
                elif 'word_embeddings' in name and 'weight' in name:
                    init_1d_row_embedding(p, pg)
                # num_class = seq_len = 512 | (512, 8)
                elif 'position_embeddings' in name and 'weight' in name:
                    init_1d_row_embedding(p, pg)
                # num_class = type_vocab_size = 2 | (2, 8)
                elif 'token_type_embeddings' in name and 'weight' in name:
                    init_1d_col_embedding(p, pg)
                elif p.process_group.tp_world_size() == 1:
                    p.set_process_group(pg)
        elif "simple_net" == model_name:
            init_spec_func(model, pg)

    model_reload = deepcopy(model)
    model = model.cuda()
    model.eval()

    model_reload = model_reload.cuda()
    model_reload.eval()

    opt_class = torch.optim.Adam
    colo_optimizer = ColossalaiOptimizer(opt_class(model.parameters(), lr=0.1))
    colo_optimizer_reload = ColossalaiOptimizer(opt_class(model_reload.parameters(), lr=0.1))

    for i, (data, label) in enumerate(train_dataloader):

        # Zero grad
        colo_optimizer.zero_grad()
        colo_optimizer_reload.zero_grad()

        data = data.to(get_current_device())
        label = label.to(get_current_device())

        dist.broadcast(data, pg.tp_rank_list()[0], pg.tp_process_group())
        dist.broadcast(label, pg.tp_rank_list()[0], pg.tp_process_group())

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            output_reload = model_reload(data)
            loss = criterion(output, label)
            loss_reload = criterion(output_reload, label)
        else:
            loss = model(data, label)
            loss_reload = model_reload(data, label)

        loss.backward()
        loss_reload.backward()

        colo_optimizer.step()
        colo_optimizer_reload.step()

        if i > 2:
            break

    if not os.path.isdir('./checkpoint') and rank == 0:
        os.mkdir('./checkpoint')
    dist.barrier()

    save_checkpoint('./checkpoint', 0, model, colo_optimizer, None)
    load_checkpoint('./checkpoint', 0, model_reload, colo_optimizer_reload, None)

    check_param_equal(model, model_reload)
    compare_optims(colo_optimizer, colo_optimizer_reload)

    if rank == 0:
        remove('./checkpoint')
    dist.barrier()


def run_dist(rank, world_size, port, use_ddp, use_mp_reload, test_scheduler):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=world_size)

    # the data loader of BERT is in DDP mode, causing the input data is not replicated in the TP context
    for model_name in ['bert']:
        _run_checkpoint(model_name,
                        init_1d_row_for_linear_weight_spec,
                        use_ddp,
                        use_mp_reload,
                        test_scheduler=test_scheduler,
                        pg=pg)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@pytest.mark.parametrize('use_ddp', [False])
@pytest.mark.parametrize('use_mp_reload', [True, False])
# @pytest.mark.parametrize('test_scheduler', ['colossalai_cosine_warmup', 'torch_cosine', 'torch_lambda'])
@rerun_if_address_is_in_use()
def test_checkpoint(world_size, use_ddp, use_mp_reload, test_scheduler=None):
    spawn(run_dist, world_size, use_ddp=use_ddp, use_mp_reload=use_mp_reload, test_scheduler=test_scheduler)


if __name__ == '__main__':
    test_checkpoint(2, use_ddp=False, use_mp_reload=True, test_scheduler="torch_cosine")

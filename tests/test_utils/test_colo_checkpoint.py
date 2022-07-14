import os, shutil
import torch
import pytest
from functools import partial

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiplicativeLR
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

import colossalai
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ComputePattern, ComputeSpec, ColoTensor, ShardSpec, ProcessGroup, DistSpecManager, ReplicaSpec
from colossalai.nn.parallel.data_parallel import ColoDDP
from colossalai.utils.checkpoint import save_checkpoint, load_checkpoint
from colossalai.nn.optimizer import ColossalaiOptimizer

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
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(torch_p, p, rtol=1e-3, atol=1e-1)


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def _run_checkpoint(model_name, init_spec_func, use_ddp, use_mp_reload, test_scheduler, pg):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)
        model_reload = model_builder(checkpoint=True)

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
                    p.redistribute(ReplicaSpec(), pg)
        elif "simple_net" == model_name:
            init_spec_func(model, pg)

    model = model.cuda()
    model.train()

    model_reload = model_reload.cuda()
    model_reload.train()

    colo_optimizer = ColossalaiOptimizer(torch.optim.SGD(model.named_parameters(), r=0.1))

    for i, (data, label) in enumerate(train_dataloader):

        # Zero grad
        colo_optimizer.zero_grad()

        data = data.to(get_current_device())
        label = label.to(get_current_device())

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        loss.backward()
        colo_optimizer.step()

        if i > 2:
            break

    if not os.path.isdir('./checkpoint') and rank == 0:
        os.mkdir('./checkpoint')
    save_checkpoint('./checkpoint', 0, model, None, None)
    dist.barrier()
    load_checkpoint('./checkpoint', 0, model_reload, None, None)

    # Since model is sharded, we merge them before param checking.
    for p in model.parameters():
        p.to_replicate_()

    for p in model_reload.parameters():
        p.to_replicate_()

    check_param_equal(model, model_reload)

    if rank == 0:
        remove('./checkpoint')


def run_dist(rank, world_size, port, use_ddp, use_mp_reload, test_scheduler):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    pg = ProcessGroup(tp_degree=world_size)
    for model_name in ['bert', 'simple_net']:
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
    run_func = partial(run_dist,
                       world_size=world_size,
                       port=free_port(),
                       use_ddp=use_ddp,
                       use_mp_reload=use_mp_reload,
                       test_scheduler=test_scheduler)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_checkpoint(2, use_ddp=False, use_mp_reload=True, test_scheduler="torch_cosine")

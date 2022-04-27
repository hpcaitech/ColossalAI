from tests.components_to_test.registry import non_distributed_component_funcs

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import named_params_with_colotensor, TensorSpec, ComputePattern, ParallelAction, ColoTensor
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc

from functools import partial
import random
import os
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_1d_row_tp():
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    set_seed(1)
    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)

    set_seed(1)
    if rank == 0:
        model_torch = model_builder(checkpoint=True)
        model_torch = model_torch.cuda()

    # A naive way to set spec for all weights in Linear
    for name, p in model.colo_named_parameters():
        if not isinstance(p, ColoTensor):
            continue
        if 'weight' in name and 'LayerNorm' not in name and 'ln' not in name and 'embed' not in name:
            p.set_spec(spec)

    model = model.cuda()

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        torch.distributed.broadcast(data, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))
        torch.distributed.broadcast(label, 0, group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Bcast rank0 data to all processes
        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        # For reference
        if rank == 0:
            if criterion:
                output_torch = model_torch(data)
                loss_torch = criterion(output_torch, label)
            else:
                output_torch = model_torch(data, label)
                loss_torch = output_torch

        if rank == 0:
            # print(loss.torch_tensor().item())
            # print('loss torch', loss_torch.item())
            assert torch.allclose(loss.torch_tensor(), loss_torch, rtol=1e-2)

        loss.backward()

        if rank == 0:
            loss_torch.backward()
        if i > 5:
            break


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_1d_row_tp()


@pytest.mark.dist
@parameterize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_simple_net(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_simple_net()

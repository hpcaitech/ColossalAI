from cProfile import label
from statistics import mode
from tests.components_to_test.registry import non_distributed_component_funcs

import colossalai
import pytest
import torch
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.utils import ColoInitContext

import torch.distributed as dist
from functools import partial


def run_simple_net():
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    with ColoInitContext():
        model = model_builder(checkpoint=True)

    # we set the Specs for weight of each linear.
    model.proj1.weight.set_spec('1Drow')
    model.proj2.weight.set_spec('1Drow')

    for i, (data, label) in enumerate(train_dataloader):
        output = model(data)
        print(output)
        if criterion:
            loss = criterion(output, label)
        else:
            loss = output

        loss.backward()

        if i > 5:
            break

    # TODO(jzy) check the results with col.nn.Linear?


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_simple_net()


@pytest.mark.dist
@parameterize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_simple_net(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_simple_net()

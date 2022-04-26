from tests.components_to_test.registry import non_distributed_component_funcs

import colossalai
import pytest
import torch.multiprocessing as mp
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import named_params_with_colotensor, TensorSpec, ComputePattern, ParallelAction, ColoTensor
from colossalai.context import ParallelMode

from functools import partial


def run_simple_net():
    # A simple net with two stacked nn.Linear
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)

    parallel_action_list = [
        ParallelAction(priority=1, compute_pattern=ComputePattern.TP1DRow, parallel_mode=ParallelMode.PARALLEL_1D)
    ]
    spec = TensorSpec(parallel_action_list)

    # A naive way to set spec for all weights in Linear
    for name, p in named_params_with_colotensor(model):
        if not isinstance(p, ColoTensor):
            continue
        if 'weight' in name and 'LayerNorm' not in name and 'ln' not in name and 'embed' not in name:
            p.set_spec(spec)

    model.cuda()

    for param in named_params_with_colotensor(model):
        print(param)

    for i, (data, label) in enumerate(train_dataloader):
        data = data.to(get_current_device())
        label = label.to(get_current_device())

        if criterion:
            output = model(data)
            loss = criterion(output, label)
        else:
            output = model(data, label)
            loss = output

        print(loss.torch_tensor())
        loss.backward()

        if i > 5:
            break

    # TODO(jzy) check the results with col.nn.Linear?


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_simple_net()


@pytest.mark.skip
@pytest.mark.dist
@parameterize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_simple_net(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_simple_net()

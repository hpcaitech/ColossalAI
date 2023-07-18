from collections import OrderedDict

import pytest
import torch

import colossalai
from colossalai.nn.parallel import ColoDDP
from colossalai.tensor import ColoParameter, ProcessGroup
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs


def check_state_dict_equal(state_dict: OrderedDict, other_state_dict: OrderedDict):
    for (k1, t1), (k2, t2) in zip(state_dict.items(), other_state_dict.items()):
        assert k1 == k2

        if t1.device != t2.device:
            temp_t2 = t2.to(t1.device)
        else:
            temp_t2 = t2

        assert torch.equal(t1, temp_t2), "\t{}\n\t{}".format(t1, temp_t2)


def init_ddp(module: torch.nn.Module) -> ColoDDP:
    pg = ProcessGroup()
    return ColoDDP(module, process_group=pg)


def run_ddp_state_dict():
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    torch_model = model_builder().cuda()
    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = init_ddp(model)
    torch_state_dict = torch_model.state_dict()

    for param in model.parameters():
        if isinstance(param, ColoParameter):
            assert param.get_process_group() is not None
    model.load_state_dict(torch_state_dict)

    for param in model.parameters():
        if isinstance(param, ColoParameter):
            assert param.get_process_group() is not None

    state_dict = model.state_dict()
    check_state_dict_equal(torch_state_dict, state_dict)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_ddp_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_state_dict(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_state_dict(2)

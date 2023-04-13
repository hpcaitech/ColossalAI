import pytest
import torch

import colossalai
from colossalai.tensor import (
    ColoParameter,
    ColoTensorSpec,
    ComputePattern,
    ComputeSpec,
    ProcessGroup,
    ReplicaSpec,
    ShardSpec,
)
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import set_seed


def run_colo_init_context(rank: int, world_size: int, port: int):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # make sure seed of each process is the same, so the params are consistent among processes and the params are exactly replicated.
    set_seed(42)
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    # keep parameters replicated during init
    with ColoInitContext(device=get_current_device()):
        model1 = model_builder()

    # shard the parameters during init
    set_seed(42)
    shard_spec = ReplicaSpec()

    # If using ShardSpec, the assertations will failed.
    # But it is not a bug, the initialized values are not consist with the original one.
    # shard_spec = ShardSpec(dims=[0], num_partitions=[world_size])
    default_pg = ProcessGroup(tp_degree=world_size)
    with ColoInitContext(device=get_current_device(), default_pg=default_pg, default_dist_spec=shard_spec):
        model2 = model_builder()

    # reshard both models
    new_shard = ShardSpec(dims=[-1], num_partitions=[world_size])
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p1: ColoParameter = p1
        p1.set_process_group(ProcessGroup(tp_degree=world_size))
        p1.set_dist_spec(new_shard)
        p2.set_dist_spec(new_shard)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert (torch.allclose(p1, p2))


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_colo_init_context(world_size):
    spawn(run_colo_init_context, world_size)


if __name__ == '__main__':
    test_colo_init_context(2)

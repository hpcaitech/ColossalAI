from functools import partial

import torch
import torch.multiprocessing as mp

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.d_tensor import DTensor, distribute_tensor
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.sharding_spec import ShardingSpec
from colossalai.utils import free_port


class TestModel(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features, out_features)
        self.linear_2 = torch.nn.Linear(out_features, in_features)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


def check_dtensor(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    test_model = TestModel(8, 8).to('cuda')
    original_tensor = torch.rand(4, 8).to('cuda')
    compare_output = test_model(original_tensor)

    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    target_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict={0: [0]})
    layout = Layout(device_mesh=device_mesh,
                    device_type=torch.device('cuda'),
                    sharding_spec=target_sharding_spec,
                    entire_shape=original_tensor.shape)
    d_tensor = DTensor(original_tensor, layout)

    assert d_tensor.entire_shape == original_tensor.shape
    assert d_tensor.data_type == original_tensor.dtype

    if rank in (0, 1):
        assert d_tensor.to_local().equal(original_tensor.narrow(0, 0, 2))
    elif rank in (2, 3):
        assert d_tensor.to_local().equal(original_tensor.narrow(0, 2, 2))
    else:
        raise ValueError(f'rank {rank} is not in the device mesh')
    assert d_tensor.to_global().equal(original_tensor)
    output = test_model(d_tensor)

    if rank in (0, 1):
        assert output.equal(compare_output.narrow(0, 0, 2))
    elif rank in (2, 3):
        assert output.equal(compare_output.narrow(0, 2, 2))
    else:
        raise ValueError(f'rank {rank} is not in the device mesh')

    new_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict={0: [0, 1]})
    new_layout = Layout(device_mesh=device_mesh,
                        device_type=torch.device('cuda'),
                        sharding_spec=new_sharding_spec,
                        entire_shape=original_tensor.shape)

    d_tensor.layout_convert(new_layout)

    if rank == 0:
        assert d_tensor.local_tensor.equal(original_tensor.narrow(0, 0, 1))
    elif rank == 1:
        assert d_tensor.local_tensor.equal(original_tensor.narrow(0, 1, 1))
    elif rank == 2:
        assert d_tensor.local_tensor.equal(original_tensor.narrow(0, 2, 1))
    elif rank == 3:
        assert d_tensor.local_tensor.equal(original_tensor.narrow(0, 3, 1))
    else:
        raise ValueError(f'rank {rank} is not in the device mesh')

    dtensor_from_local = distribute_tensor(original_tensor, new_layout)

    if rank == 0:
        assert dtensor_from_local.local_tensor.equal(original_tensor.narrow(0, 0, 1))
    elif rank == 1:
        assert dtensor_from_local.local_tensor.equal(original_tensor.narrow(0, 1, 1))
    elif rank == 2:
        assert dtensor_from_local.local_tensor.equal(original_tensor.narrow(0, 2, 1))
    elif rank == 3:
        assert dtensor_from_local.local_tensor.equal(original_tensor.narrow(0, 3, 1))
    else:
        raise ValueError(f'rank {rank} is not in the device mesh')


def test_dtensor():
    world_size = 4
    run_func = partial(check_dtensor, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_dtensor()

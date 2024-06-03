import pytest
import torch

try:
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_model

    NO_CODEGEN = False
except:
    NO_CODEGEN = True

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, rerun_if_address_is_in_use, run_on_environment_flag, spawn


class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = x * 2

        return x


class ConvModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias
        )

    def forward(self, x):
        x = self.conv(x)
        x = x * 2

        return x


def check_linear_module(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = LinearModel(4, 8).cuda()
    input = torch.rand(4, 4).cuda()
    output_compare = model(input)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    meta_args = {"x": torch.rand(4, 4).to("meta")}
    gm = initialize_model(model, meta_args=meta_args, device_mesh=device_mesh)
    output = gm(input)
    assert_close(output, output_compare)


def check_conv_module(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = ConvModel(3, 6, 2).cuda()
    input = torch.rand(4, 3, 64, 64).cuda()
    output_compare = model(input)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    meta_args = {"x": torch.rand(4, 3, 64, 64).to("meta")}
    gm = initialize_model(model, meta_args=meta_args, device_mesh=device_mesh)
    output = gm(input)
    assert_close(output, output_compare)


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.skipif(NO_CODEGEN, reason="No codegen found")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bias_addition_module():
    spawn(check_linear_module, 4)
    spawn(check_conv_module, 4)


if __name__ == "__main__":
    test_bias_addition_module()

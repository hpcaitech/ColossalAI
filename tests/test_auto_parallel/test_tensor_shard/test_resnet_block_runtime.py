from functools import partial
import pytest
import torch
import torch.multiprocessing as mp
from torch.fx import GraphModule
import torch.nn as nn
import pytest
from colossalai import device
from colossalai.initialize import launch
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.logging import disable_existing_loggers
from colossalai.auto_parallel.tensor_shard.solver.graph_analysis import GraphAnalyser
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.fx.passes.experimental.adding_shape_consistency_pass_v2 import shape_consistency_pass, solution_annotatation_pass
from colossalai.auto_parallel.tensor_shard.solver.options import SolverOptions
from colossalai.device.device_mesh import DeviceMesh
from colossalai.auto_parallel.tensor_shard.solver.strategies_constructor import StrategiesConstructor
from colossalai.auto_parallel.tensor_shard.solver.cost_graph import CostGraph
from copy import deepcopy
from colossalai.auto_parallel.tensor_shard.solver.solver import Solver
from torchvision.models import resnet34, resnet50
from colossalai.auto_parallel.tensor_shard.constants import *
from colossalai.testing import assert_close_loose, assert_close
from colossalai.testing.pytest_wrapper import run_on_environment_flag

seed = 128
cudnn_benchmark = False
cudnn_deterministic = True


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out)

        return out


def check_apply_bottleneck(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    input = torch.rand(256, 64, 64, 64).cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=False)
    entire_shape = torch.Size((4, 4, 8, 8))

    tracer = ColoTracer()
    model = Bottleneck(64, 64, 1, norm_layer=torch.nn.modules.batchnorm.BatchNorm2d).cuda()
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv1 : [#users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    #     %bn1 : [#users=1] = call_module[target=bn1](args = (%conv1,), kwargs = {})
    #     %relu : [#users=1] = call_module[target=relu](args = (%bn1,), kwargs = {})
    #     %conv2 : [#users=1] = call_module[target=conv2](args = (%relu,), kwargs = {})
    #     %bn2 : [#users=1] = call_module[target=bn2](args = (%conv2,), kwargs = {})
    #     %relu_1 : [#users=1] = call_module[target=relu](args = (%bn2,), kwargs = {})
    #     %conv3 : [#users=1] = call_module[target=conv3](args = (%relu_1,), kwargs = {})
    #     %bn3 : [#users=1] = call_module[target=bn3](args = (%conv3,), kwargs = {})
    #     %relu_2 : [#users=1] = call_module[target=relu](args = (%bn3,), kwargs = {})
    #     return relu_2
    input_sample = {'x': torch.rand(256, 64, 224, 224).to('meta')}
    cuda_rng_state = torch.cuda.get_rng_state()
    origin_output = model(input)
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    graph_analyser = GraphAnalyser(gm)
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])
    print(solution)
    device_mesh.process_groups_dict = device_mesh.create_process_groups_for_logical_mesh()
    sharding_spec_dict, origin_spec_dict = solution_annotatation_pass(gm, solution, device_mesh)
    shape_consistency_pass(gm)
    gm.recompile()
    nodes = [node for node in gm.graph.nodes]
    # TODO: wrap the gm to avoid the influence of the user training code
    torch.cuda.set_rng_state(cuda_rng_state)
    output = gm(input, sharding_spec_dict, origin_spec_dict)
    assert output.shape == origin_output.shape
    assert output.equal(origin_output)


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_apply():
    world_size = 4
    run_func = partial(check_apply_bottleneck, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_apply()

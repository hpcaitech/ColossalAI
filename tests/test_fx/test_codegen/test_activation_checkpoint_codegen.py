from operator import mod
import torch
import pytest
from torch.utils.checkpoint import checkpoint
from torch.fx import GraphModule
from colossalai.fx import ColoTracer
import colossalai
import socket

try:
    from colossalai.fx.codegen import ActivationCheckpointCodeGen
    with_codegen = True
except:
    # fall back to older pytorch version
    from colossalai.fx.codegen import python_code_with_activation_checkpoint
    with_codegen = False


def free_port(host: str = "127.0.0.1") -> int:
    """Find an available port to launch colossal (search from port 8080)

    Args:
        host: A host name, default value is 127.0.0.1

    Returns:
        An available port
    """
    # default port is 8080
    port = 8080
    while True:
        s = socket.socket()
        try:
            s.bind((host, port))
            s.close()
            return port
        except socket.error:
            pass
        port = port + 1


class MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear1(x), self.linear1(x)


class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.linear3 = torch.nn.Linear(4, 4)

    def forward(self, x):
        print(x.device)
        y1, y2 = checkpoint(self.mlp1, x)
        print(x.device)
        y3, y4 = checkpoint(self.mlp2, x)
        print(x.device)
        return y1 + y2 + y3 + y4


@pytest.mark.skipif(not with_codegen, reason='torch version is lower than 1.12.0')
def test_act_ckpt_codegen():
    # build model and run forward
    model = MyModule()
    data = torch.rand(4, 4)

    # copy model to cuda
    model = model.to(device="cuda")
    data = data.to(device="cuda")

    non_fx_out = model(data)

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)
    codegen = ActivationCheckpointCodeGen()
    graph.set_codegen(codegen)

    # check ops are annotated with ckpt
    # also annotate the selected node for offloading
    ckpt_nodes = ['mlp1_linear1', 'mlp1_linear1_1', 'mlp2_linear1', 'mlp2_linear1_1']
    offload_starts = ['mlp2_linear1']
    for node in graph.nodes:
        if node.name in ckpt_nodes:
            assert hasattr(node, 'activation_checkpoint')

            # annotate the selected node for offload
            if node.name in offload_starts:
                setattr(node, 'activation_offload', True)

    # assert checkpoint function will be generated and
    # the offload option is correct
    code = graph.python_code('self').src
    assert 'colossalai.utils.checkpoint(checkpoint_0, False, x)' in code and 'colossalai.utils.checkpoint(checkpoint_1, True, x)' in code

    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currectly
    colossalai.launch(config={}, rank=0, world_size=1, host='localhost', port=free_port(), backend='nccl')

    # recompile and verify the outputs are consistent
    gm = GraphModule(model, graph)
    gm.recompile()
    fx_out = gm(data)
    assert torch.equal(non_fx_out, fx_out)


@pytest.mark.skipif(with_codegen, reason='torch version is equal to or higher than 1.12.0')
def test_act_ckpt_python_code_torch11():
    # build model and run forward
    model = MyModule()
    data = torch.rand(4, 4)

    # copy model to cuda
    model = model.to(device="cuda")
    data = data.to(device="cuda")

    non_fx_out = model(data)

    # trace the module and replace codegen
    tracer = ColoTracer(trace_act_ckpt=True)
    graph = tracer.trace(model)

    # replace a bound method of an object
    graph._python_code = python_code_with_activation_checkpoint.__get__(graph)

    # check ops are annotated with ckpt
    ckpt_nodes = ['mlp1_linear1', 'mlp1_linear1_1', 'mlp2_linear1', 'mlp2_linear1_1']
    offload_starts = ['mlp2_linear1']
    for node in graph.nodes:
        if node.name in ckpt_nodes:
            assert hasattr(node, 'activation_checkpoint')

            # annotate the selected node for offload
            if node.name in offload_starts:
                setattr(node, 'activation_offload', True)

    # assert checkpoint function will be generated and
    # the offload option is correct
    code = graph.python_code('self').src
    assert 'colossalai.utils.checkpoint(checkpoint_0, False, x)' in code and 'colossalai.utils.checkpoint(checkpoint_1, True, x)' in code

    # launch colossalai to make sure we could execute colossalai.utils.checkpoint currectly
    colossalai.launch(config={}, rank=0, world_size=1, host='localhost', port=free_port(), backend='nccl')

    # recompile and verify the outputs are consistent
    gm = GraphModule(model, graph)
    gm.recompile()
    fx_out = gm(data)
    assert torch.equal(non_fx_out, fx_out)


if __name__ == '__main__':
    test_act_ckpt_codegen()
    test_act_ckpt_python_code_torch11()

import random

import numpy as np
import torch
from torch.fx import GraphModule

from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass
from colossalai.legacy.pipeline.middleware import Partition, Topo
from colossalai.legacy.pipeline.middleware.adaptor import get_fx_topology

MANUAL_SEED = 0
random.seed(MANUAL_SEED)
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)


class MLP(torch.nn.Module):
    def __init__(self, config={}):
        super().__init__()
        dim = config["dim"]
        layers = config["layers"]
        self.layers = torch.nn.ModuleList()

        for _ in range(layers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def split_model_and_get_DAG(model, data_gen):
    model.eval()

    # generate input sample
    kwargs = data_gen()

    # tracing model
    tracer = ColoTracer()
    try:
        meta_args = {k: v.to("meta") for k, v in kwargs.items()}
        graph = tracer.trace(root=model, meta_args=meta_args)
    except Exception as e:
        raise RuntimeError(f"Failed to trace {model.__class__.__name__}, error: {e}")
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    # apply transform passes
    annotated_model = balanced_split_pass(gm, 2)
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model)

    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, "_topo", topo)

    return top_module, split_submodules[0]._topo


def check_input(top_module, input_partition: Partition):
    partition_output = input_partition.get_output_vals()
    arg_pos = 0
    for node in top_module.graph.nodes:
        if node.op == "placeholder":
            cur_checkee = partition_output[arg_pos]
            to_partition_and_offset = cur_checkee.get()
            assert len(to_partition_and_offset) == len(node.users.keys())
            arg_pos += 1

    assert arg_pos == len(partition_output)


def check_submod(top_module, part_id, mid_partition: Partition):
    partition_input = mid_partition.get_input_vals()
    partition_output = mid_partition.get_output_vals()

    cnt = 1
    cur_node = None
    for node in top_module.graph.nodes:
        if node.name.startswith("submod"):
            cnt += 1
        if cnt == part_id:
            cur_node = node
            break

    assert len(partition_input) == len(cur_node.args)
    assert len(partition_output) == len(cur_node.users)


def check_topo(top_module, topo: Topo):
    input_partition = topo.get_input_partition()
    mid_partitions = topo.get_mid_partitions()

    check_input(top_module, input_partition)
    for part_id, submod in mid_partitions.items():
        check_submod(top_module, part_id, submod)

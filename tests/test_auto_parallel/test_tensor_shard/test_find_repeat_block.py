from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule
from transformers.pytorch_utils import Conv1D

from colossalai._analyzer.fx.passes import shape_prop_pass

# from colossalai.fx.tracer.tracer import ColoTracer
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.utils.factory import find_repeat_blocks
from colossalai.testing import clear_cache_before_run, parameterize, run_on_environment_flag

NUM_REPEAT_BLOCKS = 4
BATCH_SIZE = 1
SEQ_LENGTH = 32
HIDDEN_DIM = 384


class RepeatBlock(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.c_fc = Conv1D(intermediate_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, intermediate_size)
        self.act = torch.nn.ReLU()

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        return hidden_states


class RepeatModel(nn.Module):
    def __init__(self, intermediate_size, hidden_size, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([RepeatBlock(intermediate_size, hidden_size) for i in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class NonRepeatBlock(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_index):
        super().__init__()
        intermediate_size //= layer_index + 1
        self.c_fc = Conv1D(intermediate_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, intermediate_size)
        self.act = torch.nn.ReLU()

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)

        return hidden_states


class NonRepeatModel(nn.Module):
    def __init__(self, intermediate_size, hidden_size, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([NonRepeatBlock(intermediate_size, hidden_size, i) for i in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


@run_on_environment_flag(name="AUTO_PARALLEL")
@clear_cache_before_run()
@parameterize("model_cls", [RepeatModel, NonRepeatModel])
def test_repeat_blocks(model_cls):
    model = model_cls(4 * HIDDEN_DIM, HIDDEN_DIM, NUM_REPEAT_BLOCKS)

    tracer = ColoTracer(bias_addition_split=True)
    input_sample = {"x": torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to("meta")}
    graph = tracer.trace(root=model, meta_args=input_sample)

    gm = GraphModule(model, graph, model.__class__.__name__)
    shape_prop_pass(gm, *input_sample.values())
    gm.recompile()

    node_list = list(graph.nodes)
    root_module = graph.owning_module
    common_blocks = find_repeat_blocks(node_list, root_module, common_length_threshold=10)

    total_num_nodes = len(list(graph.nodes))
    # remove the input placeholder node and the output node
    num_repeat_nodes_per_block = (total_num_nodes - 2) // NUM_REPEAT_BLOCKS
    for common_block in common_blocks:
        print(common_block)
    if model_cls == RepeatModel:
        assert len(common_blocks) == NUM_REPEAT_BLOCKS
        assert len(common_blocks[0]) == num_repeat_nodes_per_block
    elif model_cls == NonRepeatModel:
        assert len(common_blocks) == 0


if __name__ == "__main__":
    test_repeat_blocks()

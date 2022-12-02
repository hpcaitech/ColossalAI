from typing import Optional, Tuple, Union

import torch
# from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import torch.nn as nn
import transformers
from torch.fx import GraphModule
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers.pytorch_utils import Conv1D

from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
from colossalai.auto_parallel.tensor_shard.solver import (
    CostGraph,
    GraphAnalyser,
    Solver,
    SolverOptions,
    StrategiesConstructor,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.testing import parameterize
from colossalai.testing.pytest_wrapper import run_on_environment_flag

BATCH_SIZE = 1
SEQ_LENGTH = 32
HIDDEN_DIM = 768


# The reason Why we don't import GPT2Attention from transformers directly is that:
# 1. The tracer will not work correctly when we feed meta_args and concrete_args at same time,
# so we have to build the customized GPT2Attention class and remove the conditional branch manually.
# 2. The order of split and view op has been changed in the customized GPT2Attention class, the new
# order is same as megatron-lm gpt model.
class GPT2Attention(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions),
                                  dtype=torch.uint8)).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1)**0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length].to(torch.bool)
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)    # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        qkv = self.c_attn(hidden_states)

        # query = self._split_heads(query, self.num_heads, self.head_dim)
        # key = self._split_heads(key, self.num_heads, self.head_dim)
        # value = self._split_heads(value, self.num_heads, self.head_dim)
        query, key, value = self._split_heads(qkv, self.num_heads, 3 * self.head_dim).split(self.head_dim, dim=3)
        present = (key, value)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs    # a, present, (attentions)


class GPT2Block(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        # %transformer_h_0_ln_1
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_output = attn_outputs[0]    # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + outputs[1:]

        return outputs    # hidden_states, present, (attentions, cross_attentions)


@run_on_environment_flag(name='AUTO_PARALLEL')
@parameterize('model_cls', [GPT2Block, GPT2Attention, GPT2MLP])
def test_self_attention_block(model_cls):
    config = transformers.GPT2Config(n_position=64, n_layer=4, n_head=16, n_embd=HIDDEN_DIM)
    if model_cls == GPT2MLP:
        model = model_cls(intermediate_size=4 * config.hidden_size, config=config)
    else:
        model = model_cls(config=config)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    if model_cls == GPT2MLP:
        input_sample = {
            'hidden_states': torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to('meta'),
        }
    else:
        input_sample = {
            'hidden_states': torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to('meta'),
            'attention_mask': torch.rand(1, SEQ_LENGTH).to('meta'),
        }

    graph = tracer.trace(root=model, meta_args=input_sample)

    gm = GraphModule(model, graph, model.__class__.__name__)
    print(gm.graph)
    gm.recompile()
    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser, memory_budget=-1)
    ret = solver.call_solver_serialized_args()
    strategies_list = solver.last_s_val
    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]

    computation_cost = 0
    communication_cost = 0
    memory_cost = 0
    for index, node in enumerate(nodes):
        print(node.name, node.strategies_vector[strategies_list[index]].name)
        computation_cost += node.strategies_vector[strategies_list[index]].compute_cost.total
        communication_cost += node.strategies_vector[strategies_list[index]].communication_cost.total
        node_memory_cost = node.strategies_vector[strategies_list[index]].memory_cost.total
        if isinstance(node_memory_cost, tuple):
            node_memory_cost = node_memory_cost[0]
        memory_cost += node_memory_cost.activation + node_memory_cost.parameter

    print(f'computation cost is {computation_cost}')
    print(f'communication cost is {communication_cost}')
    print(f'memory cost is {memory_cost}')


if __name__ == '__main__':
    test_self_attention_block()

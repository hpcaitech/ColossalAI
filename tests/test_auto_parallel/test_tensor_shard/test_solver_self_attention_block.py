from typing import Optional, Tuple, Union

import torch
# from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import torch.nn as nn
import transformers
from torch.fx import GraphModule
from torchvision.models import resnet50
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

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
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
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
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

        if not self.is_cross_attention:
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
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)    # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`.")

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            qkv = self.c_attn(hidden_states)

        # query = self._split_heads(query, self.num_heads, self.head_dim)
        # key = self._split_heads(key, self.num_heads, self.head_dim)
        # value = self._split_heads(value, self.num_heads, self.head_dim)
        query, key, value = self._split_heads(qkv, self.num_heads, 3 * self.head_dim).split(self.head_dim, dim=3)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value)

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs    # a, present, (attentions)


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_self_attention_block():
    config = transformers.GPT2Config(n_position=64, n_layer=4, n_head=16, n_embd=HIDDEN_DIM)
    model_cls = GPT2Attention
    model = model_cls(config=config)
    # output = model(torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM),  attention_mask=torch.rand(1, SEQ_LENGTH))
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
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

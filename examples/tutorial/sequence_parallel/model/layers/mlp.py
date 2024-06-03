import torch.nn as nn
import torch.nn.functional as F

from colossalai.kernel.jit import bias_gelu_impl

from .linear import Linear


class TransformerMLP(nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, hidden_size, mlp_ratio, fuse_gelu=True):
        super(TransformerMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = Linear(hidden_size, int(hidden_size * mlp_ratio), skip_bias_add=True)

        self.bias_gelu_fusion = fuse_gelu
        self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = Linear(int(hidden_size * mlp_ratio), hidden_size, skip_bias_add=True)

    def forward(self, hidden_states):
        # hidden states should be in the shape of [s, b, h]
        # it will be projects into [s, b, 4h]
        # and projected back to [s, b, h]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

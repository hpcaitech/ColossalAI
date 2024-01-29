import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_func.cross_entropy import vocab_cross_entropy

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.nn.layer.layernorm import MixedFusedLayerNorm as LayerNorm

from .linear import Linear
from .pooler import Pooler


class BertLMHead(nn.Module):
    """Masked LM head for Bert
    Arguments:
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
    ):
        super(BertLMHead, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

        self.dense = Linear(hidden_size, hidden_size)
        self.layernorm = LayerNorm(hidden_size)
        self.gelu = torch.nn.functional.gelu

    def forward(self, hidden_states, word_embeddings_weight, lm_labels):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)

        output = F.linear(hidden_states, word_embeddings_weight, self.bias)
        lm_loss = vocab_cross_entropy(output, lm_labels)

        return lm_loss


class BertBinaryHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pooler = Pooler(hidden_size)
        self.dense = Linear(hidden_size, 2)

    def forward(self, hidden_states):
        if gpc.get_local_rank(ParallelMode.SEQUENCE) == 0:
            output = self.pooler(hidden_states)
            output = self.dense(output)
        else:
            output = None
        return output


class BertDualHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, add_binary_head):
        super().__init__()
        self.lm_head = BertLMHead(vocab_size, hidden_size)
        self.add_binary_head = add_binary_head
        if add_binary_head:
            self.binary_head = BertBinaryHead(hidden_size)
        else:
            self.binary_head = None

    def forward(self, hidden_states, word_embeddings_weight, lm_labels):
        if self.add_binary_head:
            binary_output = self.binary_head(hidden_states)
        else:
            binary_output = None
        lm_loss = self.lm_head(hidden_states, word_embeddings_weight, lm_labels)
        return lm_loss, binary_output

import torch

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn import TransformerSelfAttentionRing
from colossalai.utils import get_current_device


def check_selfattention():
    WORLD_SIZE = gpc.get_world_size(ParallelMode.SEQUENCE)
    SUB_SEQ_LENGTH = 8
    BATCH = 4
    HIDDEN_SIZE = 16

    layer = TransformerSelfAttentionRing(16, 8, 8, 0.1)
    layer = layer.to(get_current_device())

    hidden_states = torch.rand(SUB_SEQ_LENGTH, BATCH, HIDDEN_SIZE).to(get_current_device())
    attention_mask = torch.randint(low=0, high=2,
                                   size=(BATCH, 1, 1, 1, SUB_SEQ_LENGTH * WORLD_SIZE)).to(get_current_device())
    out = layer(hidden_states, attention_mask)

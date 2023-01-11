import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LayerNorm

from .kernel import bias_dropout_add
from .ops import SelfAttention, Transition


class MSARowAttentionWithPairBias(nn.Module):

    def __init__(self, d_node, d_pair, c=32, n_head=8, p_drop=0.15):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.d_node = d_node
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernormM = LayerNorm(d_node)
        self.layernormZ = LayerNorm(d_pair)

        _init_weights = torch.nn.init.normal_(torch.zeros([n_head, d_pair]),
                                              std=1.0 / math.sqrt(d_pair))
        self.linear_b_weights = nn.parameter.Parameter(data=_init_weights, requires_grad=True)

        self.attention = SelfAttention(qkv_dim=d_node,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_node,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_node,)), requires_grad=True)

    def forward(self, M_raw, Z):
        ## Input projections
        M = self.layernormM(M_raw)
        Z = self.layernormZ(Z)
        b = F.linear(Z, self.linear_b_weights)
        b = b.permute(0, 3, 1, 2)
        # b = rearrange(b, 'b q k h -> b h q k')

        M = self.attention(M, b)
        dropout_mask = torch.ones_like(M[:, 0:1, :, :]).to(M.device).to(M.dtype)

        return bias_dropout_add(M, self.out_bias, dropout_mask, M_raw, prob=self.p_drop)


class MSAColumnAttention(nn.Module):

    def __init__(self, d_node, c=32, n_head=8):
        super(MSAColumnAttention, self).__init__()
        self.d_node = d_node
        self.c = c
        self.n_head = n_head

        self.layernormM = LayerNorm(d_node)
        self.attention = SelfAttention(qkv_dim=d_node,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_node,
                                       gating=True)

    def forward(self, M_raw):
        M = M_raw.transpose(-2, -3)
        M = self.layernormM(M)

        M = self.attention(M)

        M = M.transpose(-2, -3)
        return M_raw + M


class MSAStack(nn.Module):

    def __init__(self, d_node, d_pair, p_drop=0.15):
        super(MSAStack, self).__init__()

        self.MSARowAttentionWithPairBias = MSARowAttentionWithPairBias(d_node=d_node,
                                                                       d_pair=d_pair,
                                                                       p_drop=p_drop)

        self.MSAColumnAttention = MSAColumnAttention(d_node=d_node)
        self.MSATransition = Transition(d=d_node)

    def forward(self, node, pair):
        node = self.MSARowAttentionWithPairBias(node, pair)
        node = self.MSAColumnAttention(node)
        node = self.MSATransition(node)

        return node

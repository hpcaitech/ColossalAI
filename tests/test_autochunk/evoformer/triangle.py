import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from .kernel import bias_dropout_add, bias_ele_dropout_residual
from .ops import Linear, SelfAttention, Transition


def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class TriangleMultiplicationOutgoing(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(TriangleMultiplicationOutgoing, self).__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_projection = Linear(d_pair, c)
        self.right_projection = Linear(d_pair, c)
        self.left_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)
        self.right_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(d_pair, d_pair, initializer='zeros', use_bias=False)
        self.output_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

        self.p_drop = p_drop

    def forward(self, Z_raw):
        Z = self.layernorm1(Z_raw)
        left_proj_act = self.left_projection(Z)
        right_proj_act = self.right_projection(Z)

        left_proj_act = left_proj_act * torch.sigmoid(self.left_gate(Z))
        right_proj_act = right_proj_act * torch.sigmoid(self.right_gate(Z))

        g = torch.sigmoid(self.output_gate(Z))
        # p = torch.matmul(
        #     permute_final_dims(left_proj_act, (2, 0, 1)),
        #     permute_final_dims(right_proj_act, (2, 1, 0)),
        # )
        # ab = permute_final_dims(p, (1, 2, 0))

        ab = torch.einsum('bikd,bjkd->bijd', left_proj_act, right_proj_act)
        ab = self.output_projection(self.layernorm2(ab))
        dropout_mask = torch.ones_like(Z[:, 0:1, :, :]).to(Z.device).to(Z.dtype)
        return bias_ele_dropout_residual(ab,
                                         self.output_bias,
                                         g,
                                         dropout_mask,
                                         Z_raw,
                                         prob=self.p_drop)


class TriangleMultiplicationIncoming(nn.Module):

    def __init__(self, d_pair, p_drop, c=128):
        super(TriangleMultiplicationIncoming, self).__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_projection = Linear(d_pair, c)
        self.right_projection = Linear(d_pair, c)
        self.left_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)
        self.right_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(d_pair, d_pair, initializer='zeros', use_bias=False)
        self.output_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

        self.p_drop = p_drop

    def forward(self, Z_raw):
        Z = self.layernorm1(Z_raw)
        left_proj_act = self.left_projection(Z)
        right_proj_act = self.right_projection(Z)

        left_proj_act = left_proj_act * torch.sigmoid(self.left_gate(Z))
        right_proj_act = right_proj_act * torch.sigmoid(self.right_gate(Z))

        g = torch.sigmoid(self.output_gate(Z))
        # p = torch.matmul(
        #     permute_final_dims(left_proj_act, (2, 1, 0)),
        #     permute_final_dims(right_proj_act, (2, 0, 1)),
        # )
        # ab = permute_final_dims(p, (1, 2, 0))

        ab = torch.einsum('bkid,bkjd->bijd', left_proj_act, right_proj_act)
        ab = self.output_projection(self.layernorm2(ab))
        dropout_mask = torch.ones_like(Z[:, 0:1, :, :]).to(Z.device).to(Z.dtype)
        return bias_ele_dropout_residual(ab,
                                         self.output_bias,
                                         g,
                                         dropout_mask,
                                         Z_raw,
                                         prob=self.p_drop)


class TriangleAttentionStartingNode(nn.Module):

    def __init__(self, d_pair, p_drop, c=32, n_head=4):
        super(TriangleAttentionStartingNode, self).__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernorm1 = LayerNorm(d_pair)
        _init_weights = torch.nn.init.normal_(torch.zeros([d_pair, n_head]),
                                              std=1.0 / math.sqrt(d_pair))
        self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)
        self.attention = SelfAttention(qkv_dim=d_pair,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_pair,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

    def forward(self, Z_raw):
        Z = self.layernorm1(Z_raw)
        b = torch.einsum('bqkc,ch->bhqk', Z, self.linear_b_weights)

        Z = self.attention(Z, b)

        dropout_mask = torch.ones_like(Z[:, 0:1, :, :]).to(Z.device).to(Z.dtype)
        return bias_dropout_add(Z, self.out_bias, dropout_mask, Z_raw, prob=self.p_drop)


class TriangleAttentionEndingNode(nn.Module):

    def __init__(self, d_pair, p_drop, c=32, n_head=4):
        super(TriangleAttentionEndingNode, self).__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head
        self.p_drop = p_drop

        self.layernorm1 = LayerNorm(d_pair)
        _init_weights = torch.nn.init.normal_(torch.zeros([d_pair, n_head]),
                                              std=1.0 / math.sqrt(d_pair))
        self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)
        self.attention = SelfAttention(qkv_dim=d_pair,
                                       c=c,
                                       n_head=n_head,
                                       out_dim=d_pair,
                                       gating=True,
                                       last_bias_fuse=True)

        self.out_bias = nn.parameter.Parameter(data=torch.zeros((d_pair,)), requires_grad=True)

    def forward(self, Z_raw):
        Z = Z_raw.transpose(-2, -3)
        Z = self.layernorm1(Z)
        b = torch.einsum('bqkc,ch->bhqk', Z, self.linear_b_weights)

        Z = self.attention(Z, b)

        Z = Z.transpose(-2, -3)
        dropout_mask = torch.ones_like(Z[:, :, 0:1, :]).to(Z.device).to(Z.dtype)
        return bias_dropout_add(Z, self.out_bias, dropout_mask, Z_raw, prob=self.p_drop)


class PairStack(nn.Module):

    def __init__(self, d_pair, p_drop=0.25):
        super(PairStack, self).__init__()

        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(d_pair, p_drop=p_drop)
        self.TriangleMultiplicationIncoming = TriangleMultiplicationIncoming(d_pair, p_drop=p_drop)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(d_pair, p_drop=p_drop)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(d_pair, p_drop=p_drop)
        self.PairTransition = Transition(d=d_pair)

    def forward(self, pair):
        pair = self.TriangleMultiplicationOutgoing(pair)
        pair = self.TriangleMultiplicationIncoming(pair)
        pair = self.TriangleAttentionStartingNode(pair)
        pair = self.TriangleAttentionEndingNode(pair)
        pair = self.PairTransition(pair)
        return pair

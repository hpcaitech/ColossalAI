import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LayerNorm

from .initializer import glorot_uniform_af
from .kernel import bias_sigmod_ele


class DropoutRowwise(nn.Module):

    def __init__(self, p):
        super(DropoutRowwise, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        dropout_mask = torch.ones_like(x[:, 0:1, :, :])
        dropout_mask = self.dropout(dropout_mask)
        return dropout_mask * x


class DropoutColumnwise(nn.Module):

    def __init__(self, p):
        super(DropoutColumnwise, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        dropout_mask = torch.ones_like(x[:, :, 0:1, :])
        dropout_mask = self.dropout(dropout_mask)
        return dropout_mask * x


class Transition(nn.Module):

    def __init__(self, d, n=4):
        super(Transition, self).__init__()
        self.norm = LayerNorm(d)
        self.linear1 = Linear(d, n * d, initializer='relu')
        self.linear2 = Linear(n * d, d, initializer='zeros')

    def forward(self, src):
        x = self.norm(src)
        x = self.linear2(F.relu(self.linear1(x)))
        return src + x


class OutProductMean(nn.Module):

    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=32):
        super(OutProductMean, self).__init__()

        self.layernormM = LayerNorm(n_feat)
        self.linear_a = Linear(n_feat, n_feat_proj)
        self.linear_b = Linear(n_feat, n_feat_proj)

        self.o_linear = Linear(n_feat_proj * n_feat_proj,
                               n_feat_out,
                               initializer='zero',
                               use_bias=True)

    def forward(self, M):
        M = self.layernormM(M)
        left_act = self.linear_a(M)
        right_act = self.linear_b(M)

        O = torch.einsum('bsid,bsje->bijde', left_act, right_act).contiguous()
        # O = rearrange(O, 'b i j d e -> b i j (d e)')
        O = O.reshape(O.shape[0], O.shape[1], O.shape[2], -1)
        Z = self.o_linear(O)

        return Z


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        feature_in: int,
        feature_out: int,
        initializer: str = 'linear',
        use_bias: bool = True,
        bias_init: float = 0.,
    ):
        super(Linear, self).__init__(feature_in, feature_out, bias=use_bias)

        self.use_bias = use_bias
        if initializer == 'linear':
            glorot_uniform_af(self.weight, gain=1.0)
        elif initializer == 'relu':
            glorot_uniform_af(self.weight, gain=2.0)
        elif initializer == 'zeros':
            nn.init.zeros_(self.weight)
        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(bias_init)


class SelfAttention(nn.Module):
    """
    Multi-Head SelfAttention dealing with [batch_size1, batch_size2, len, dim] tensors
    """

    def __init__(self, qkv_dim, c, n_head, out_dim, gating=True, last_bias_fuse=False):
        super(SelfAttention, self).__init__()
        self.qkv_dim = qkv_dim
        self.c = c
        self.n_head = n_head
        self.out_dim = out_dim
        self.gating = gating
        self.last_bias_fuse = last_bias_fuse

        self.scaling = self.c**(-0.5)

        # self.to_qkv = Linear(qkv_dim, 3 * n_head * c, initializer='linear')
        self.to_q = Linear(qkv_dim, n_head * c, initializer='linear', use_bias=False)
        self.to_k = Linear(qkv_dim, n_head * c, initializer='linear', use_bias=False)
        self.to_v = Linear(qkv_dim, n_head * c, initializer='linear', use_bias=False)

        if gating:
            self.gating_bias = nn.parameter.Parameter(data=torch.ones((n_head * c,)))
            self.gating_linear = Linear(qkv_dim, n_head * c, initializer='zero', use_bias=False)

        self.o_linear = Linear(n_head * c,
                               out_dim,
                               initializer='zero',
                               use_bias=(not last_bias_fuse))

    def forward(self, in_data, nonbatched_bias=None):
        """
        :param in_data: [batch_size1, batch_size2, len_qkv, qkv_dim]
        :param bias: None or [batch_size1, batch_size2, n_head, len_q, len_kv]
        :param nonbatched_bias: None or [batch_size1, n_head, len_q, len_kv]
        """

        # qkv = self.to_qkv(in_data).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head), qkv)

        q = self.to_q(in_data)
        k = self.to_k(in_data)
        v = self.to_v(in_data)

        # q, k, v = map(lambda t: rearrange(t, 'b1 b2 n (h d) -> b1 b2 h n d', h=self.n_head),
        #               [q, k, v])
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], t.shape[2], self.n_head, -1).permute(0, 1, 3, 2, 4),
                      [q, k, v])
        
        q = q * self.scaling

        logits = torch.matmul(q, k.transpose(-1, -2))

        if nonbatched_bias is not None:
            logits += nonbatched_bias.unsqueeze(1)
        weights = torch.softmax(logits, dim=-1)
        # weights = softmax(logits)

        weighted_avg = torch.matmul(weights, v)
        # weighted_avg = rearrange(weighted_avg, 'b1 b2 h n d -> b1 b2 n (h d)')
        weighted_avg = weighted_avg.permute(0, 1, 3, 2, 4)
        weighted_avg = weighted_avg.reshape(weighted_avg.shape[0], weighted_avg.shape[1], weighted_avg.shape[2], -1)

        if self.gating:
            gate_values = self.gating_linear(in_data)
            weighted_avg = bias_sigmod_ele(gate_values, self.gating_bias, weighted_avg)

        output = self.o_linear(weighted_avg)
        return output

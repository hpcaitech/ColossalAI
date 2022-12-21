import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
from math import log2, floor

def exists(val):
    return val is not None

# normalization

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# AliBi

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_sim):
        h, i, j, device = *qk_sim.shape[-3:], qk_sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent=False)

        return bias

# residual

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = RMSNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5

        self.alibi_pos_biases = AlibiPositionalBias(heads = self.heads)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask

        self.register_buffer("mask", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), 1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def forward(self, x):

        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys or values (shared key / values is a personal discovery of mine), and feedforward inner

        q, kv, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h = h)

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, kv)

        # add the alibi bias

        sim = sim + self.alibi_pos_biases(sim)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, kv)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        merge_heads = self.attn_out(out) + self.ff_out(ff)
        return merge_heads

def PaLM(*, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4):

    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),
        *[Residual(ParallelTransformerBlock(dim, dim_head, heads, ff_mult)) for _ in range(depth)],
        RMSNorm(dim),
        nn.Linear(dim, num_tokens, bias=False)
    )

    # they used embedding weight tied projection out to logits, not common, but works
    
    net[-1].weight = net[0].weight

    nn.init.normal_(net[0].weight, std=0.02)
    return net

# For testing functionality of the model

if __name__ == "__main__":

    palm = PaLM(
        num_tokens = 20000,
        dim = 512,
        depth = 1,
        heads = 8,
        dim_head = 64,
    )

    tokens = torch.randint(0, 20000, (1, 2048))
    logits = palm(tokens) # (1, 2048, 20000)

    n_params_torch = sum(
        p.numel() for p in palm.parameters() if p.requires_grad
    )

    print(f"Number of parameters in torch model: {n_params_torch}")

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import matmul, nn

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# parallel with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelResidual(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return x + sum([fn(x) for fn in self.fns])


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device)
        # freqs = einsum("i , j -> i j", seq.type_as(self.inv_freq), self.inv_freq)
        # freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
        i, j = len(seq.type_as(self.inv_freq)), len(self.inv_freq)
        freqs = matmul(seq.type_as(self.inv_freq).reshape(i, 1), self.inv_freq.reshape(1, j))
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# feedforward
# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# attention
class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("position", pos_emb, persistent=False)
        return pos_emb

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

        # queries, keys, values

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        b, h, i, d, j = q.size(0), q.size(1), q.size(2), q.size(3), k.size(1)

        # similarity

        # sim = einsum("b h i d, b j d -> b h i j", q, k)
        sim = matmul(q.reshape(b, h * i, d), k.transpose(1, 2))
        sim = sim.reshape(b, h, i, j)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        b_, h_, i_, j_, d_ = attn.size(0), attn.size(1), attn.size(2), attn.size(3), v.size(2)

        # aggregate values

        # out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = matmul(attn.reshape(b_, h_ * i_, j_), v)
        out = out.reshape(b_, h_, i_, d_)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# transformer


def PaLM(*, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4):
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),
        *[
            ParallelResidual(
                Attention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            )
            for _ in range(depth)
        ],
        LayerNorm(dim),
        nn.Linear(dim, num_tokens, bias=False),
    )

    # they used embedding weight tied projection out to logits, not common, but works
    net[-1].weight = net[0].weight

    nn.init.normal_(net[0].weight, std=0.02)
    return net

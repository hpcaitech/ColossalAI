import math
import torch
import torch.nn as nn
from colossalai.context import ParallelMode
from colossalai.nn.layer import VanillaPatchEmbedding, VanillaClassifier, \
    WrappedDropout as Dropout, WrappedDropPath as DropPath
from colossalai.nn.layer.moe import Experts, MoeLayer, Top2Router, NormalNoiseGenerator
from .util import moe_sa_args, moe_mlp_args
from ..helper import TransformerLayer
from colossalai.global_variables import moe_env
from colossalai.utils import get_current_device


class VanillaSelfAttention(nn.Module):
    """Standard ViT self attention.
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_kv: int,
                 attention_drop: float = 0,
                 drop_rate: float = 0,
                 bias: bool = True,
                 dropout1=None,
                 dropout2=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.scale = 1.0 / math.sqrt(self.d_kv)

        self.dense1 = nn.Linear(d_model, 3 * n_heads * d_kv, bias, device=get_current_device())
        self.softmax = nn.Softmax(dim=-1)
        self.atten_drop = nn.Dropout(attention_drop) if dropout1 is None else dropout1
        self.dense2 = nn.Linear(n_heads * d_kv, d_model, device=get_current_device())
        self.dropout = nn.Dropout(drop_rate) if dropout2 is None else dropout2

    def forward(self, x):
        qkv = self.dense1(x)
        new_shape = qkv.shape[:2] + (3, self.n_heads, self.d_kv)
        qkv = qkv.view(*new_shape)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[:]

        x = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        x = self.atten_drop(self.softmax(x))

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_shape = x.shape[:2] + (self.n_heads * self.d_kv,)
        x = x.reshape(*new_shape)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


class VanillaFFN(nn.Module):
    """FFN composed with two linear layers, also called MLP.
    """
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 activation=None,
                 drop_rate: float = 0,
                 bias: bool = True,
                 dropout1=None,
                 dropout2=None):
        super().__init__()
        dense1 = nn.Linear(d_model, d_ff, bias, device=get_current_device())
        act = nn.GELU() if activation is None else activation
        dense2 = nn.Linear(d_ff, d_model, bias, device=get_current_device())
        drop1 = nn.Dropout(drop_rate) if dropout1 is None else dropout1
        drop2 = nn.Dropout(drop_rate) if dropout2 is None else dropout2

        self.ffn = nn.Sequential(
            dense1, act, drop1,dense2, drop2)

    def forward(self, x):
        return self.ffn(x)


class Widenet(nn.Module):
    def __init__(self,
                 num_experts: int,
                 capacity_factor: float,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 depth: int = 12,
                 d_model: int = 768,
                 num_heads: int = 12,
                 d_kv: int = 64,
                 d_ff: int = 3072,
                 attention_drop: float = 0.,
                 drop_rate: float = 0.1,
                 drop_path: float = 0.):
        super().__init__()

        embedding = VanillaPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_size=d_model)
        embed_dropout = Dropout(p=drop_rate, mode=ParallelMode.TENSOR)

        shared_sa = VanillaSelfAttention(**moe_sa_args(
            d_model=d_model, n_heads=num_heads, d_kv=d_kv,
            attention_drop=attention_drop, drop_rate=drop_rate))

        noisy_func = NormalNoiseGenerator(num_experts)
        shared_router = Top2Router(capacity_factor, noisy_func=noisy_func)
        shared_experts = Experts(expert=VanillaFFN,
                                 num_experts=num_experts,
                                 **moe_mlp_args(
                                     d_model=d_model,
                                     d_ff=d_ff,
                                     drop_rate=drop_rate
                                 ))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        blocks = [
            TransformerLayer(
                att=shared_sa,
                ffn=MoeLayer(dim_model=d_model, num_experts=num_experts,
                             router=shared_router, experts=shared_experts),
                norm1=nn.LayerNorm(d_model, eps=1e-6),
                norm2=nn.LayerNorm(d_model, eps=1e-6),
                droppath=DropPath(p=dpr[i], mode=ParallelMode.TENSOR)
            )
            for i in range(depth)
        ]
        norm = nn.LayerNorm(d_model, eps=1e-6)
        self.linear = VanillaClassifier(in_features=d_model,
                                        num_classes=num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.widenet = nn.Sequential(embedding, embed_dropout, *blocks, norm)

    def forward(self, x):
        moe_env.reset_loss()
        x = self.widenet(x)
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        return x

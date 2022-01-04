import torch
import torch.nn as nn
from colossalai.context import ParallelMode
from colossalai.nn.layer import VanillaPatchEmbedding, VanillaSelfAttention, \
    VanillaFFN, VanillaClassifier, WrappedDropout as Dropout, WrappedDropPath as DropPath
from colossalai.nn.layer.moe import Experts, MoeLayer, Top2Router, NormalNoiseGenerator
from .util import moe_sa_args, moe_mlp_args
from ..helper import TransformerLayer
from colossalai.global_variables import moe_env


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
                SA=shared_sa,
                FFN=MoeLayer(dim_model=d_model, num_experts=num_experts,
                             router=shared_router, experts=shared_experts),
                NORM1=nn.LayerNorm(d_model, eps=1e-6),
                NORM2=nn.LayerNorm(d_model, eps=1e-6),
                DROPPATH=DropPath(p=dpr[i], mode=ParallelMode.TENSOR)
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

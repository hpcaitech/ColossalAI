import torch
import torch.nn as nn
from siu.fx import symbolic_trace
from siu.fx.symbolic_trace import register_leaf_module, register_leaf_module_impl

from colossalai.kernel.cuda_native.flash_attention import FlashAttention
from colossalai.kernel.cuda_native.layer_norm import MixedFusedLayerNorm

register_leaf_module(FlashAttention)
register_leaf_module(MixedFusedLayerNorm)


@register_leaf_module_impl(FlashAttention)
def flash_attn_shape_impl(qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
    if key_padding_mask is None:
        return qkv.new_empty((qkv.shape[0], qkv.shape[1], qkv.shape[3], qkv.shape[4])), None
    else:
        raise NotImplementedError


@register_leaf_module_impl(MixedFusedLayerNorm)
def mixed_fused_ln_shape_impl(x: torch.Tensor):
    return x.new_empty(x.shape)


class MyModule(nn.Module):

    def __init__(self, normalized_shape=(128, 8, 64)) -> None:
        super().__init__()
        self.attn = FlashAttention()
        self.ln = MixedFusedLayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float16)
        for i in range(x.shape[0]):
            x, _ = self.attn(x)
            new_shape = x.shape[:2] + (-1,) + x.shape[2:]
            x = x.repeat(1, 3, 1, 1).reshape(new_shape)
        return x


qkv = torch.rand(3, 128, 3, 8, 64).cuda()
m = MyModule().cuda()
gm = symbolic_trace(m, meta_args=dict(x=qkv))
gm(qkv)

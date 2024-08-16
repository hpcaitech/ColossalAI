import math
from copy import copy

import torch
from torch.testing import assert_close

from colossalai.kernel.kernel_loader import FlashAttentionLoader, FlashAttentionWithCustomMaskLoader
from colossalai.shardformer.layer import AttnMaskType, ColoAttention
from colossalai.shardformer.layer.attn import invert_mask
from colossalai.testing import clear_cache_before_run, parameterize
from colossalai.utils import get_current_device, set_seed

DTYPE = [torch.float16, torch.bfloat16]
B, N, S, D = 2, 8, 256, 32

TOL_MAP = {
    torch.float16: {"atol": 5e-4, "rtol": 2e-3},
    torch.bfloat16: {},
}


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None, dropout_p=0.0):
    head_dim = q.size(-1)
    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float).to(q.dtype)
    attn_weights = torch.dropout(attn_weights, p=dropout_p, train=True)
    attn_output = torch.matmul(attn_weights, v)
    return attn_output


def gen_padded_kwargs(dtype: torch.dtype):
    padding_mask = torch.ones((B, S), dtype=torch.int, device=get_current_device())
    padding_mask[0, : S // 4] = 0
    return (
        ColoAttention.prepare_attn_kwargs((B, 1, S, S), dtype, padding_mask.device, q_padding_mask=padding_mask),
        padding_mask,
    )


def gen_padded_causal_kwargs(dtype: torch.dtype):
    padding_mask = torch.ones((B, S), dtype=torch.int, device=get_current_device())
    padding_mask[0, S // 2 :] = 0
    return (
        ColoAttention.prepare_attn_kwargs(
            (B, 1, S, S), dtype, padding_mask.device, q_padding_mask=padding_mask, is_causal=True
        ),
        padding_mask,
    )


def gen_causal_kwargs(dtype: torch.dtype):
    return ColoAttention.prepare_attn_kwargs((B, 1, S, S), dtype, get_current_device(), is_causal=True), None


def gen_custom_kwargs(dtype: torch.dtype):
    attn_mask = torch.ones((B, S, S), dtype=dtype, device=get_current_device())
    attn_mask[0, : S // 2, S // 2 :] = 0
    attn_mask[0, S // 2 :, : S // 2] = 0
    attn_mask[1, :, S // 4 :] = 0
    attn_mask = invert_mask(attn_mask).unsqueeze(1)
    assert not torch.all(attn_mask != 0, dim=-1).any()
    return {"attention_mask": attn_mask}, None


def post_process_kwargs_for_raw_attn(attn_kwargs: dict):
    if "attention_mask_type" in attn_kwargs:
        attn_kwargs = copy(attn_kwargs)
        mask_type = attn_kwargs.pop("attention_mask_type")
        attn_kwargs["is_causal"] = mask_type in (AttnMaskType.CAUSAL, AttnMaskType.PADDED_CAUSAL)
    return attn_kwargs


def check_attn_func(dtype: torch.dtype, attn_func, attn_kwargs: dict, padding_mask=None):
    tols = TOL_MAP[dtype]
    q = torch.rand((B, N, S, D), dtype=dtype, device=get_current_device(), requires_grad=True)
    k = torch.rand((B, N, S, D), dtype=dtype, device=get_current_device(), requires_grad=True)
    v = torch.rand((B, N, S, D), dtype=dtype, device=get_current_device(), requires_grad=True)
    q_flash = q.clone().detach().requires_grad_(True)
    k_flash = k.clone().detach().requires_grad_(True)
    v_flash = v.clone().detach().requires_grad_(True)
    attn_mask = attn_kwargs.get("attention_mask", None)
    ref_output = attention_ref(q, k, v, attn_mask)
    output = attn_func(q_flash, k_flash, v_flash, **attn_kwargs)
    if padding_mask is not None:
        # [B, Sq] -> [B, 1, Sq, 1]
        padding_mask = padding_mask[:, None, :, None].logical_not()
        ref_output = ref_output.masked_fill(padding_mask, 0)
        output = output.masked_fill(padding_mask, 0)

    assert_close(output, ref_output, **tols)
    output.mean().backward()
    ref_output.mean().backward()
    assert_close(q.grad, q_flash.grad, **tols)
    assert_close(k.grad, k_flash.grad, **tols)
    assert_close(v.grad, v_flash.grad, **tols)


@clear_cache_before_run()
@parameterize("dtype", DTYPE)
def test_flash_attn_func(dtype: torch.dtype):
    torch.backends.cudnn.deterministic = True
    set_seed(0)
    # (func, name, need_postprocess)
    avail_attn_funcs = [(ColoAttention.attention, "coloattn", False)]
    avail_custom_mask_attn_funcs = [(ColoAttention.attention, "coloattn", False)]
    avail_padding_mask_attn_funcs = [(ColoAttention.attention, "coloattn", False)]
    for ext_cls in FlashAttentionLoader.REGISTRY:
        ext = ext_cls()
        if ext.is_available():
            ext.assert_compatible()
            avail_attn_funcs.append((ext.load(), ext.name, True))
    for ext_cls in FlashAttentionWithCustomMaskLoader.REGISTRY:
        ext = ext_cls()
        if ext.is_available():
            ext.assert_compatible()
            avail_custom_mask_attn_funcs.append((ext.load(), ext.name, True))

    test_sets = {
        "none": (lambda dtype: ({}, None), avail_attn_funcs),
        "padded": (gen_padded_kwargs, avail_padding_mask_attn_funcs),
        "padded_causal": (gen_padded_causal_kwargs, avail_padding_mask_attn_funcs),
        "causal": (gen_causal_kwargs, avail_attn_funcs),
        "custom": (gen_custom_kwargs, avail_custom_mask_attn_funcs),
    }

    for mask_type, (gen_kwargs_func, attn_funcs) in test_sets.items():
        attn_kwargs, padding_mask = gen_kwargs_func(dtype)
        for attn_func, name, need_postprocess in attn_funcs:
            print(f"{dtype}, {name}, {mask_type}")
            if mask_type == "padded":
                pass
            if need_postprocess:
                check_attn_func(dtype, attn_func, post_process_kwargs_for_raw_attn(attn_kwargs), padding_mask)
            else:
                check_attn_func(dtype, attn_func, attn_kwargs, padding_mask)


if __name__ == "__main__":
    test_flash_attn_func()

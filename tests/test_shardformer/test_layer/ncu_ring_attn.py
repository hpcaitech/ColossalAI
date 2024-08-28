import torch
from flash_attn import flash_attn_qkvpacked_func

bs, seq_len, nheads, d = 4, 4096, 32, 128
qkv = torch.randn(bs, seq_len, 3, nheads, d, device="cuda:0", dtype=torch.bfloat16)
out, lse, _ = flash_attn_qkvpacked_func(qkv, causal=True, return_attn_probs=True)

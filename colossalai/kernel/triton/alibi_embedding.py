from typing import Optional 

import torch
import triton
import triton.languaga as tl

"""
# Base autotune if needed
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":4,},num_warps=4),
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":8,},num_warps=8),
        triton.Config({'BLOCK_HEAD':8,"BLOCK_TOKENS":8,},num_warps=8),
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":4,},num_warps=16),
        triton.Config({'BLOCK_HEAD':4,"BLOCK_TOKENS":4,},num_warps=32),
        triton.Config({'BLOCK_HEAD':16,"BLOCK_TOKENS":16,},num_warps=4),
        triton.Config({'BLOCK_HEAD':8,"BLOCK_TOKENS":16,},num_warps=8),
    ],
    key=['HEAD_DIM','q_total_tokens','Q_HEAD_NUM']
)
"""

@triton.jit
def flash_attn_with_alibi {
    
}
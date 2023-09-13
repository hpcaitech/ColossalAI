import math

import numpy as np
import torch
from torch.nn import functional as F


def torch_context_attention(xq, xk, xv, bs, seqlen, num_head, head_dim):
    '''
     adepted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/bloom/triton_kernel/context_flashattention_nopad.py#L253
    '''
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    sm_scale = 1 / math.sqrt(head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) * sm_scale
    scores = F.softmax(scores.float() + mask, dim=-1).to(dtype=torch.float16)

    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output

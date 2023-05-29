import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Model
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder

from .gpt_attention import XGPT2Attention, XGPT2Model
from .opt_attention import XOPTAttention, XOPTDecoder


def wrap_attention(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, GPT2Model):
            module.__class__ = XGPT2Model
        elif isinstance(module, GPT2Attention):
            module.__class__ = XGPT2Attention
        elif isinstance(module, OPTAttention):
            module.__class__ = XOPTAttention
        elif isinstance(module, OPTDecoder):
            module.__class__ = XOPTDecoder
    return model

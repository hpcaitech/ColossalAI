import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTAttention

from .opt_attn import XOPTAttention


def convert_to_xformer_model(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, OPTAttention):
            module.__class__ = XOPTAttention
    return model


def recover_from_xformer_model(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, XOPTAttention):
            module.__class__ = OPTAttention
    return model

import torch
import torch.nn as nn

from .model_utils import find_layers
from .quant import make_quant


def load_quant(model: nn.Module, checkpoint: str, wbits: int, groupsize: int):
    model = model.eval()
    layers = find_layers(model)

    # ignore lm head
    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]

    make_quant(model, layers, wbits, groupsize)

    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    return model

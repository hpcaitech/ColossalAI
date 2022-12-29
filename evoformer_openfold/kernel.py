import torch
import torch.nn.functional as F


def bias_sigmod_ele(y, bias, z):
    return torch.sigmoid(y + bias) * z


def bias_dropout_add(x: torch.Tensor, bias: torch.Tensor, dropmask: torch.Tensor,
                     residual: torch.Tensor, prob: float) -> torch.Tensor:
    out = (x + bias) * F.dropout(dropmask, p=prob, training=False)
    out = residual + out
    return out


def bias_ele_dropout_residual(ab: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
                              dropout_mask: torch.Tensor, Z_raw: torch.Tensor,
                              prob: float) -> torch.Tensor:
    return Z_raw + F.dropout(dropout_mask, p=prob, training=True) * (g * (ab + b))
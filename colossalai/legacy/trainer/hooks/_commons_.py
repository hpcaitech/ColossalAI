import torch


def _format_number(val, prec=5):
    if isinstance(val, float):
        return f"{val:.{prec}g}"
    elif torch.is_tensor(val) and torch.is_floating_point(val):
        return f"{val.item():.{prec}g}"
    return val

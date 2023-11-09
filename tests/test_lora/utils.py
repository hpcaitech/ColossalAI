from typing import Callable, List, Optional

import torch
from torch import nn

from colossalai.booster import Booster
from colossalai.testing import assert_equal, assert_not_equal


def do_fwd_bwd(
    booster: Booster,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_gen_fn: Callable,
    output_transform_fn: Callable,
    criterion: Callable,
):
    for _ in range(2):
        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        output = model(**data)
        output = output_transform_fn(output)
        loss = criterion(output)

        booster.backward(loss, optimizer)
        optimizer.clip_grad_by_norm(1.0)
        optimizer.step()


def check_param_equality(name: str, p1: nn.Parameter, p2: nn.Parameter, modules_to_save: Optional[List[str]]):
    p2 = p2.to(p1.device).to(p1.dtype)
    if "lora_" in name:
        # lora modules should be updated
        assert_not_equal(p1, p2)
    else:
        if (modules_to_save is not None) and any(f"{key}.modules_to_save" in name for key in modules_to_save):
            # if a non-lora module should be saved, it should be updated
            assert_not_equal(p1, p2)
        else:
            # if a non-lora module isn't supposed to be saved, it shouldn't be updated
            assert_equal(p1, p2)

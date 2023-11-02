from copy import deepcopy
from typing import Type, Union

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from colossalai.nn.optimizer import CPUAdam, FusedAdam, HybridAdam
from tests.kit.model_zoo import model_zoo

_ALLOWED_OPTIM_DEVICES = [
    (FusedAdam, torch.device("cuda:0")),
    (CPUAdam, torch.device("cpu")),
    (CPUAdam, torch.device("cuda:0")),
    (HybridAdam, torch.device("cpu")),
    (HybridAdam, torch.device("cuda:0")),
]

_ALLOWED_P_G_TYPES = [
    (torch.float, torch.float),  # pure fp32
    (torch.float, torch.half),  # fp16 amp
    (torch.float, torch.bfloat16),  # bfloat16 amp
]

N_STEPS = 3


def setup_param_groups(bert_model: nn.Module) -> list:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def set_grad(model: nn.Module, torch_model: nn.Module, g_dtype: torch.dtype) -> None:
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        torch_p.grad = torch.rand_like(torch_p)
        # avoid inconsistent grad and param dtype error
        orig_p = p.data
        p.data = torch_p.grad.clone().to(g_dtype)
        p.grad = p.data
        p.data = orig_p


@pytest.mark.parametrize("optim_cls, device", _ALLOWED_OPTIM_DEVICES)
@pytest.mark.parametrize("adamw", [False, True])
@pytest.mark.parametrize("p_dtype, g_dtype", _ALLOWED_P_G_TYPES)
def test_adam_optim_on_bert(
    optim_cls: Union[Type[FusedAdam], Type[CPUAdam], Type[HybridAdam]],
    device: torch.device,
    adamw: bool,
    p_dtype: torch.dtype,
    g_dtype: torch.dtype,
) -> None:
    model_fn, *_ = next(iter(model_zoo.get_sub_registry("transformers_bert_for_sequence_classification").values()))
    torch_model = model_fn().to(device)
    model = deepcopy(torch_model).to(p_dtype)
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    torch_optim_cls = AdamW if adamw else Adam
    torch_optim = torch_optim_cls(setup_param_groups(torch_model), lr=lr, betas=(beta1, beta2), eps=eps)
    optim = optim_cls(setup_param_groups(model), lr=lr, betas=(beta1, beta2), eps=eps, adamw_mode=adamw)

    rtol, atol = 1e-5, 1e-5
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 2e-3, 2e-3
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 4e-3, 4e-3

    for _ in range(N_STEPS):
        set_grad(model, torch_model, g_dtype)
        torch_optim.step()
        optim.step()
        torch_optim.zero_grad()
        optim.zero_grad()
        for p, torch_p in zip(model.parameters(), torch_model.parameters()):
            # if overflow, the weight won't be updated. so there will be no nan in p
            assert not torch.isnan(p).any()
            assert torch.allclose(p.float(), torch_p, rtol=rtol, atol=atol)

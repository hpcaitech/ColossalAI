"""Optimizer steps on fixed weights and gradients, for comparing an optimizer's update rule with a reference one.

Plain PyTorch, no plugin: the gradients are given, so only the update is measured.
"""

from typing import Callable, Dict, Iterable, Optional, Union

import torch
from torch import Tensor, nn

from ..trace import Trace


def run_optimizer(
    optimizer_fn: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
    params: Union[nn.Module, Dict[str, Tensor]],
    dtype: torch.dtype,
    steps: int = 1,
    grad_seed: int = 0,
    grads_fn: Optional[Callable[[int, str, Tensor], Tensor]] = None,
    **meta,
) -> Trace:
    """Run ``steps`` steps of ``optimizer_fn`` on copies of ``params`` (a module's parameters or a name to tensor
    dict) cast to ``dtype``, and record ``param/<step>/<name>`` after every step.

    The gradient of each parameter at step t is ``grads_fn(t, name, weight)`` if given, else a standard normal tensor
    seeded by ``grad_seed + t`` (parameters in name order); it is produced in float64 and cast to ``dtype``, so every
    run sees the same gradients up to that cast. The optimizer is built on the parameters in name order.
    """
    if isinstance(params, nn.Module):
        params = dict(params.named_parameters())
    names = sorted(params)
    weights = {n: nn.Parameter(params[n].detach().clone().to(dtype)) for n in names}
    optimizer = optimizer_fn([weights[n] for n in names])
    results = {}
    for step in range(steps):
        generator = torch.Generator(device="cpu").manual_seed(grad_seed + step)
        for name in names:
            weight = weights[name]
            if grads_fn is not None:
                grad = grads_fn(step, name, weight.detach()).to(torch.float64)
            else:
                grad = torch.randn(weight.shape, generator=generator, dtype=torch.float64)
            weight.grad = grad.to(device=weight.device, dtype=dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        results.update({f"param/{step}/{n}": weights[n].detach().clone() for n in names})
    return Trace.capture(
        results,
        dtype=str(dtype).replace("torch.", ""),
        optimizer=type(optimizer).__qualname__,
        steps=steps,
        grad_seed=grad_seed,
        **meta,
    )

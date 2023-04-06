import math

import torch
import torch.nn as nn
from numpy import dtype

from colossalai.testing import clear_cache_before_run, parameterize
from colossalai.utils import multi_tensor_applier


def torch_adam_update(
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    use_adamw,
):
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    if weight_decay != 0:
        if use_adamw:
            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)
        else:
            grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    param.addcdiv_(exp_avg, denom, value=-step_size)


@clear_cache_before_run()
@parameterize('adamw', [False, True])
@parameterize('step', [1, 2])
@parameterize('p_dtype', [torch.float, torch.half])
@parameterize('g_dtype', [torch.float, torch.half])
def test_adam(adamw, step, p_dtype, g_dtype):
    from colossalai.kernel.op_builder import FusedOptimBuilder
    fused_optim = FusedOptimBuilder().load()
    fused_adam = fused_optim.multi_tensor_adam

    dummy_overflow_buf = torch.cuda.IntTensor([0])

    count = 0

    for i in range(3):
        p = torch.rand(64, dtype=p_dtype).cuda()
        p_copy = p.clone().float()
        g = torch.rand(p.shape, dtype=g_dtype).cuda()
        g_copy = g.clone().float()
        m = torch.rand(p.shape).cuda()
        m_copy = m.clone()
        v = torch.rand(p.shape).cuda()
        v_copy = v.clone()

        lr = 1e-3
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        weight_decay = 0

        multi_tensor_applier(fused_adam, dummy_overflow_buf, [[g], [p], [m], [v]], lr, beta1, beta2, eps, step, adamw,
                             True, weight_decay, -1)

        torch_adam_update(
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            p_copy,    # fp32 data
            g_copy,    # fp32 grad
            m_copy,
            v_copy,
            adamw,
        )

        if torch.isnan(p).any() or torch.isnan(p_copy).any():
            count += 1
            continue
        assert count < 200, "too many nans"
        assert torch.allclose(p.to(torch.float), p_copy.to(torch.float), 1e-5,
                              1e-5), f"failed check, adamw {adamw}, p_dtype {p_dtype}, g_dtype {g_dtype}"

import math

import torch

from colossalai.testing import clear_cache_before_run, parameterize


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


def assertLess(data_diff, threshold, msg):
    assert data_diff < threshold, msg


def assertTrue(condition, msg):
    assert condition, msg


@clear_cache_before_run()
@parameterize('adamw', [True, False])
@parameterize('step', [1, 2])
@parameterize('p_dtype', [torch.float, torch.half])
@parameterize('g_dtype', [torch.float, torch.half])
def test_cpu_adam(adamw, step, p_dtype, g_dtype):
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    weight_decay = 0

    for i in range(3):
        p_data = torch.rand(64, dtype=p_dtype)
        p_data_copy = p_data.clone().float()
        p_grad = torch.rand(64, dtype=g_dtype)
        p_grad_copy = p_grad.clone().float()
        exp_avg = torch.rand(p_data.shape)
        exp_avg_copy = exp_avg.clone()
        exp_avg_sq = torch.rand(p_data.shape)
        exp_avg_sq_copy = exp_avg_sq.clone()

        from colossalai.kernel.op_builder import CPUAdamBuilder
        cpu_optim = CPUAdamBuilder().load()

        cpu_adam_op = cpu_optim.CPUAdamOptimizer(lr, beta1, beta2, eps, weight_decay, adamw)

        cpu_adam_op.step(
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            True,
            p_data.view(-1),    # fp32 data
            p_grad.view(-1),    # fp32 grad
            exp_avg.view(-1),
            exp_avg_sq.view(-1),
            -1,
        )

        torch_adam_update(
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            p_data_copy,    # fp32 data
            p_grad_copy,    # fp32 grad
            exp_avg_copy,
            exp_avg_sq_copy,
            adamw,
        )
        var = p_data_copy - p_data
        data_diff = torch.max(torch.abs(var))
        threshold = 1e-3
        assertLess(
            data_diff,
            threshold,
            f"p_data diff {data_diff}. failed check, step {step}, lr {lr}, eps "
            f"{eps} beta1 {beta1} beta2 {beta2} weight_decay {weight_decay} p_dtype {p_dtype}, g_dtype {g_dtype}",
        )
        max_grad_diff = torch.max(torch.abs(p_grad_copy - p_grad))
        assertTrue(max_grad_diff < threshold, f"diff {max_grad_diff}")
        max_exp_avg_diff = torch.max(torch.abs(exp_avg_copy - exp_avg))
        assertTrue(max_exp_avg_diff < threshold, f"max_exp_avg_diff {max_exp_avg_diff}")
        max_exp_avg_sq_diff = torch.max(torch.abs(exp_avg_sq_copy - exp_avg_sq))
        assertTrue(max_exp_avg_sq_diff < threshold, f"max_exp_avg_sq_diff {max_exp_avg_sq_diff}")


if __name__ == '__main__':
    test_cpu_adam()

# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import torch
try:
    import cpu_adam
except ImportError:
    raise ImportError("import cpu_adam error")


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
    loss_scale,
    use_adamw,
):
    if loss_scale > 0:
        grad.div_(loss_scale)
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


class Test():

    def __init__(self):
        self.opt_id = 0

    def assertLess(self, data_diff, threshold, msg):
        assert data_diff < threshold, msg

    def assertTrue(self, condition, msg):
        assert condition, msg

    def check_res(
        self,
        step,
        lr,
        eps,
        beta1,
        beta2,
        weight_decay,
        shape,
        grad_dtype,
        loss_scale,
        use_adamw,
        cpu_adam_op,
    ):
        p_data = torch.rand(shape, dtype=grad_dtype)
        p_data_copy = p_data.clone().float()
        p_grad = torch.rand(shape, dtype=grad_dtype)
        if loss_scale > 0:
            p_grad.mul_(loss_scale)
        p_grad_copy = p_grad.clone().float()
        exp_avg = torch.rand(shape)
        exp_avg_copy = exp_avg.clone()
        exp_avg_sq = torch.rand(shape)
        exp_avg_sq_copy = exp_avg_sq.clone()

        cpu_adam_op.create_adam(0, lr, beta1, beta2, eps, weight_decay, use_adamw, True)
        cpu_adam_op.adam_update(
            self.opt_id,
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
            loss_scale,
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
            loss_scale,
            use_adamw,
        )

        if loss_scale > 0:
            p_grad.div_(loss_scale)

        var = p_data_copy - p_data
        data_diff = torch.max(torch.abs(var))
        threshold = 2e-3 if grad_dtype else 1e-4
        self.assertLess(
            data_diff,
            threshold,
            f"p_data diff {data_diff}. failed check, step {step}, lr {lr} eps "
            f"{eps} beta1 {beta1} beta2 {beta2} weight_decay {weight_decay} loss_scale {loss_scale} grad_dtype {grad_dtype}",
        )
        max_grad_diff = torch.max(torch.abs(p_grad_copy - p_grad))
        self.assertTrue(max_grad_diff < threshold, f"diff {max_grad_diff}")
        max_exp_avg_diff = torch.max(torch.abs(exp_avg_copy - exp_avg))
        self.assertTrue(max_exp_avg_diff < threshold, f"max_exp_avg_diff {max_exp_avg_diff}")
        max_exp_avg_sq_diff = torch.max(torch.abs(exp_avg_sq_copy - exp_avg_sq))
        self.assertTrue(max_exp_avg_sq_diff < threshold, f"max_exp_avg_sq_diff {max_exp_avg_sq_diff}")

    def test_cpu_adam(self):
        lr = 0.9
        eps = 1e-6
        weight_decay = 0
        for use_adamw in [False, True]:
            for shape in [(23,), (8, 24)]:
                for step in range(1, 2):
                    for lr in [0.01]:
                        for eps in [1e-8]:
                            for beta1 in [0.9]:
                                for beta2 in [0.999]:
                                    for weight_decay in [0.001]:
                                        for grad_dtype in [torch.half, torch.float]:
                                            for loss_scale in [-1, 2**5]:
                                                self.check_res(
                                                    step,
                                                    lr,
                                                    eps,
                                                    beta1,
                                                    beta2,
                                                    weight_decay,
                                                    shape,
                                                    grad_dtype,
                                                    loss_scale,
                                                    use_adamw,
                                                    cpu_adam,
                                                )


def test_cpu_adam():
    test_case = Test()
    test_case.test_cpu_adam()


if __name__ == "__main__":
    test = Test()
    test.test_cpu_adam()

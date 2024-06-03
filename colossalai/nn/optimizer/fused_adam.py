# modified from https://github.com/NVIDIA/apex/blob/master/apex/optimizers/fused_adam.py
"""
Copyright 2020 The Microsoft DeepSpeed Team

Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit a109f85
Licensed under the MIT License.
"""
import torch

from colossalai.utils import get_current_device, multi_tensor_applier


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    `FusedAdam` requires CUDA extensions which can be built during installation or runtime.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`colossalai.nn.optimizer.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adamw_mode=False``

    :class:`colossalai.nn.optimizer.FusedAdam` may be used with or without Amp.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adamw_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adamw_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        set_grad_none=True,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adamw_mode = 1 if adamw_mode else 0
        self.set_grad_none = set_grad_none
        if multi_tensor_applier.available:
            from colossalai.kernel.kernel_loader import FusedOptimizerLoader

            fused_optim = FusedOptimizerLoader().load()

            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=get_current_device())
            self.multi_tensor_adam = fused_optim.multi_tensor_adam
        else:
            raise RuntimeError("FusedAdam requires cuda extensions")

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, div_scale: float = -1):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                "FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments."
            )
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            # create lists for multi-tensor apply
            g_l, p_l, m_l, v_l = [], [], [], []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                if p.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
                    raise RuntimeError("FusedAdam only support fp16, fp32 and bf16.")

                g_l.append(p.grad.data)
                p_l.append(p.data)
                m_l.append(state["exp_avg"])
                v_l.append(state["exp_avg_sq"])

            multi_tensor_applier(
                self.multi_tensor_adam,
                self._dummy_overflow_buf,
                [g_l, p_l, m_l, v_l],
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                group["step"],
                self.adamw_mode,
                bias_correction,
                group["weight_decay"],
                div_scale,
            )

        return loss

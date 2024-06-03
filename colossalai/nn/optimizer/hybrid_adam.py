from typing import Any, Optional

import torch

from colossalai.kernel.kernel_loader import FusedOptimizerLoader
from colossalai.utils import get_current_device, multi_tensor_applier

from .cpu_adam import CPUAdam


class HybridAdam(CPUAdam):
    """Implements Adam algorithm.

    Supports parameters updating on both GPU and CPU, depending on the device of parameters.
    But the parameters and gradients should on the same device:
      * Parameters on CPU and gradients on CPU is allowed.
      * Parameters on GPU and gradients on GPU is allowed.
      * Parameters on GPU and gradients on CPU is **not** allowed.

    `HybridAdam` requires CUDA extensions which can be built during installation or runtime.

    This version of Hybrid Adam is an hybrid of CPUAdam and FusedAdam.

    * For parameters updating on CPU, it uses CPUAdam.
    * For parameters updating on GPU, it uses FusedAdam.
    * Hybrid precision calculation of fp16 and fp32 is supported, eg fp32 parameters and fp16 gradients.

    :class:`colossalai.nn.optimizer.HybridAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adamw_mode=False``

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        model_params (iterable): iterable of parameters of dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED yet in CPUAdam!
        adamw_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        simd_log (boolean, optional): whether to show if you are using SIMD to
            accelerate. (default: False)
        nvme_offload_fraction (float, optional): Fraction of optimizer states to be offloaded to NVMe. Defaults to 0.0.
        nvme_offload_dir (Optional[str], optional): Directory to save NVMe offload files.
            If it's ``None``, a random temporary directory will be used. Defaults to None.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    # Number of fp32 shards for per parameter
    # Param weight, grad, momentum and variance
    num_fp32_shards_per_param = 4

    def __init__(
        self,
        model_params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adamw_mode=True,
        nvme_offload_fraction: float = 0.0,
        nvme_offload_dir: Optional[str] = None,
        **defaults: Any,
    ):
        super().__init__(
            model_params,
            lr,
            bias_correction,
            betas,
            eps,
            weight_decay,
            adamw_mode,
            nvme_offload_fraction,
            nvme_offload_dir,
        )
        if torch.cuda.is_available():
            fused_optim = FusedOptimizerLoader().load()
            self.gpu_adam_op = fused_optim.multi_tensor_adam
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=get_current_device())

    @torch.no_grad()
    def step(self, closure=None, div_scale: float = -1):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._pre_step("exp_avg", "exp_avg_sq")
        for _, group in enumerate(self.param_groups):
            g_l, p_l, m_l, v_l = [], [], [], []
            group_step = 0
            for _, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]

                target_device = p.device
                if len(state) == 0:
                    state["step"] = 0
                    # gradient momentums
                    state["exp_avg"] = torch.zeros_like(p, device=target_device)
                    # gradient variances
                    state["exp_avg_sq"] = torch.zeros_like(p, device=target_device)
                    self._post_state_init(p)

                state["step"] += 1
                group_step = state["step"]
                beta1, beta2 = group["betas"]

                if target_device.type == "cpu" or target_device.type == "npu":
                    assert state["exp_avg"].device.type in ("cpu", "npu"), "exp_avg should stay on cpu"
                    assert state["exp_avg_sq"].device.type in ("cpu", "npu"), "exp_avg should stay on cpu"
                    self._pre_update(p, "exp_avg", "exp_avg_sq")
                    if p.grad.dtype is torch.bfloat16 or p.grad.device.type == "npu":
                        # cpu adam kernel does not support bf16 now
                        bias_correction1 = 1 - beta1 ** state["step"]
                        bias_correction2 = 1 - beta2 ** state["step"]
                        self.torch_adam_update(
                            p.data,
                            p.grad.data,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            group["lr"],
                            beta1,
                            beta2,
                            group["eps"],
                            group["weight_decay"],
                            bias_correction1,
                            bias_correction2,
                            self.adamw_mode,
                        )
                    else:
                        self.cpu_adam_op.step(
                            state["step"],
                            group["lr"],
                            beta1,
                            beta2,
                            group["eps"],
                            group["weight_decay"],
                            group["bias_correction"],
                            p.data,
                            p.grad.data,
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            div_scale,
                        )
                    self._post_update(p, "exp_avg", "exp_avg_sq")

                elif target_device.type == "cuda":
                    assert state["exp_avg"].device.type == "cuda", "exp_avg should stay on cuda"
                    assert state["exp_avg_sq"].device.type == "cuda", "exp_avg should stay on cuda"

                    # record the state by group and update at once
                    g_l.append(p.grad.data)
                    p_l.append(p.data)
                    m_l.append(state["exp_avg"])
                    v_l.append(state["exp_avg_sq"])

                else:
                    raise RuntimeError
            if len(g_l) > 0:
                adamw_mode = 1 if self.adamw_mode else 0
                bias_correction = 1 if group["bias_correction"] else 0
                multi_tensor_applier(
                    self.gpu_adam_op,
                    self._dummy_overflow_buf,
                    [g_l, p_l, m_l, v_l],
                    group["lr"],
                    group["betas"][0],
                    group["betas"][1],
                    group["eps"],
                    group_step,
                    adamw_mode,
                    bias_correction,
                    group["weight_decay"],
                    div_scale,
                )
        self._post_step()
        return loss

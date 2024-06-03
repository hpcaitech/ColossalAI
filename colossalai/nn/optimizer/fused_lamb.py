# modified from https://github.com/NVIDIA/apex/blob/master/apex/optimizers/fused_lamb.py
import torch

from colossalai.utils import multi_tensor_applier


class FusedLAMB(torch.optim.Optimizer):
    """Implements LAMB algorithm.

    `FusedLAMB` requires CUDA extensions which can be built during installation or runtime.

    This version of fused LAMB implements 2 fusions.

      * Fusion of the LAMB update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`colossalai.nn.optimizer.FusedLAMB`'s usage is identical to any ordinary Pytorch optimizer

    :class:`colossalai.nn.optimizer.FusedLAMB` may be used with or without Amp.

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        use_nvlamb (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        amsgrad=False,
        adam_w_mode=True,
        grad_averaging=True,
        set_grad_none=True,
        max_grad_norm=1.0,
        use_nvlamb=False,
    ):
        if amsgrad:
            raise RuntimeError("FusedLAMB does not support the AMSGrad variant.")
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
        )
        super(FusedLAMB, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            from colossalai.kernel.kernel_loader import FusedOptimizerLoader

            fused_optim = FusedOptimizerLoader().load()

            self.multi_tensor_l2norm = fused_optim.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor(
                [0], dtype=torch.int, device=self.param_groups[0]["params"][0].device
            )
            self.multi_tensor_lamb = fused_optim.multi_tensor_lamb
        else:
            raise RuntimeError("FusedLAMB requires cuda extensions")

        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group["params"]:
                    p.grad = None
        else:
            super(FusedLAMB, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError("FusedLAMB only support fp16 and fp32.")

        device = self.param_groups[0]["params"][0].device
        g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(1, device=device)
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [g_all_32], False)[0]
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(self.multi_tensor_l2norm, self._dummy_overflow_buf, [g_all_16], False)[0]

        # blend two grad norms to get global grad norm
        global_grad_norm = multi_tensor_applier(
            self.multi_tensor_l2norm, self._dummy_overflow_buf, [[g_norm_32, g_norm_16]], False
        )[0]
        max_grad_norm = self.defaults["max_grad_norm"]

        for group in self.param_groups:
            bias_correction = 1 if group["bias_correction"] else 0
            beta1, beta2 = group["betas"]
            grad_averaging = 1 if group["grad_averaging"] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "FusedLAMB does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state["exp_avg"])
                    v_16.append(state["exp_avg_sq"])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state["exp_avg"])
                    v_32.append(state["exp_avg_sq"])
                else:
                    raise RuntimeError("FusedLAMB only support fp16 and fp32.")

            if len(g_16) > 0:
                multi_tensor_applier(
                    self.multi_tensor_lamb,
                    self._dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    bias_correction,
                    group["weight_decay"],
                    grad_averaging,
                    self.adam_w_mode,
                    global_grad_norm,
                    max_grad_norm,
                    self.use_nvlamb,
                )
            if len(g_32) > 0:
                multi_tensor_applier(
                    self.multi_tensor_lamb,
                    self._dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["step"],
                    bias_correction,
                    group["weight_decay"],
                    grad_averaging,
                    self.adam_w_mode,
                    global_grad_norm,
                    max_grad_norm,
                    self.use_nvlamb,
                )

        return loss

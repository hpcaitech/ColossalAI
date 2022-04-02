import torch

from colossalai.utils import multi_tensor_applier
from colossalai.registry import OPTIMIZERS
from colossalai.nn.optimizer import CPU_ADAM_CNT


@OPTIMIZERS.register_module
class HybridAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Supports parameters updating on both GPU and CPU, depanding on the device of paramters.
    But the parameters and gradients should on the same device: 
      * Parameters on CPU and gradients on CPU is allowed.
      * Parameters on GPU and gradients on GPU is allowed.
      * Parameters on GPU and gradients on CPU is **not** allowed.

    Requires ColossalAI to be installed via ``pip install .``

    This version of Hybrid Adam is an hybrid of CPUAdam and FusedAdam.

    * For parameters updating on CPU, it uses CPUAdam.
    * For parameters updating on GPU, it uses FusedAdam.
    * Hybird precision calculation of fp16 and fp32 is supported, eg fp32 parameters and fp16 gradients.

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

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    # Number of fp32 shards for per parameter
    # Param weight, grad, momentum and variance
    num_fp32_shards_per_param = 4

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 adamw_mode=True,
                 simd_log=False):

        default_args = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super(HybridAdam, self).__init__(model_params, default_args)
        self.opt_id = CPU_ADAM_CNT()
        self.adamw_mode = adamw_mode
        try:
            import cpu_adam
            import colossal_C
        except ImportError:
            raise ImportError('Please install colossalai from source code to use HybridAdam')

        self.cpu_adam_op = cpu_adam
        self.cpu_adam_op.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, simd_log)

        self.gpu_adam_op = colossal_C.multi_tensor_adam
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])

    def __del__(self):
        if self.cpu_adam_op:
            self.cpu_adam_op.destroy_adam(self.opt_id)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for _, group in enumerate(self.param_groups):
            g_l, p_l, m_l, v_l = [], [], [], []
            group_step = 0
            for _, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                state = self.state[p]

                target_device = p.device
                if len(state) == 0:
                    state['step'] = 0

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float, device=target_device)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float, device=target_device)

                state['step'] += 1
                group_step = state['step']
                beta1, beta2 = group['betas']

                if target_device.type == 'cpu':
                    assert state['exp_avg'].device.type == 'cpu', "exp_avg should stay on cpu"
                    assert state['exp_avg_sq'].device.type == 'cpu', "exp_avg should stay on cpu"
                    self.cpu_adam_op.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                                 group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                                 state['exp_avg'], state['exp_avg_sq'], -1)

                elif target_device.type == 'cuda':
                    assert state['exp_avg'].device.type == 'cuda', "exp_avg should stay on cuda"
                    assert state['exp_avg_sq'].device.type == 'cuda', "exp_avg should stay on cuda"

                    # record the state by gruop and update at once
                    g_l.append(p.grad.data)
                    p_l.append(p.data)
                    m_l.append(state['exp_avg'])
                    v_l.append(state['exp_avg_sq'])

                else:
                    raise RuntimeError
            if len(g_l) > 0:
                adamw_mode = 1 if self.adamw_mode else 0
                bias_correction = 1 if group['bias_correction'] else 0
                multi_tensor_applier(self.gpu_adam_op, self._dummy_overflow_buf, [g_l, p_l, m_l, v_l], group['lr'],
                                     group['betas'][0], group['betas'][1], group['eps'], group_step, adamw_mode,
                                     bias_correction, group['weight_decay'])
        return loss

import math
import torch

from colossalai.registry import OPTIMIZERS


@OPTIMIZERS.register_module
class CPUAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Supports parameters updating on both GPU and CPU, depanding on the device of paramters.
    But the parameters and gradients should on the same device: 
      * Parameters on CPU and gradients on CPU is allowed.
      * Parameters on GPU and gradients on GPU is allowed.
      * Parameters on GPU and gradients on CPU is **not** allowed.

    Requires ColossalAI to be installed via ``pip install .``.

    This version of CPU Adam accelates parameters updating on CPU with SIMD.
    Support of AVX2 or AVX512 is required.

    The GPU part is implemented in an naive way.

    CPU Adam also supports the hybrid precision calculation, eg. fp32 parameters and fp16 gradients.

    :class:`colossalai.nn.optimizer.CPUAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
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
    
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    optimizer_id = 0
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
        super(CPUAdam, self).__init__(model_params, default_args)
        self.opt_id = CPUAdam.optimizer_id
        CPUAdam.optimizer_id = CPUAdam.optimizer_id + 1
        self.adamw_mode = adamw_mode
        try:
            import cpu_adam
        except ImportError:
            raise ImportError('Please install colossalai from source code to use CPUAdam')
        self.cpu_adam_op = cpu_adam
        self.cpu_adam_op.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, simd_log)

    def __del__(self):
        if self.cpu_adam_op:
            self.cpu_adam_op.destroy_adam(self.opt_id)

    def torch_adam_update(self,
                          data,
                          grad,
                          exp_avg,
                          exp_avg_sq,
                          lr,
                          beta1,
                          beta2,
                          eps,
                          weight_decay,
                          bias_correction1,
                          bias_correction2,
                          use_adamw=False):
        # FIXME(ver217): remove the below line when replace torch adam with fused adam
        grad = grad.float()

        if weight_decay != 0:
            if use_adamw:
                data.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(data, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # TODO(jiaruifang) dose not support amsgrad
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for _, group in enumerate(self.param_groups):
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

                    bias_correction1 = 1 - beta1**state['step']
                    bias_correction2 = 1 - beta2**state['step']

                    # adam on cuda
                    self.torch_adam_update(p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'], group['lr'],
                                           beta1, beta2, group['eps'], group['weight_decay'], bias_correction1,
                                           bias_correction2, self.adamw_mode)
                else:
                    raise RuntimeError
        return loss

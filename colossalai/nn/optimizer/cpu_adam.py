import torch


class CPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-8,
                 weight_decay=0,
                 adamw_mode=True,
                 loss_scale=-1,
                 simd_log=False):

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction)
        super(CPUAdam, self).__init__(model_params, default_args)
        self.opt_id = CPUAdam.optimizer_id
        CPUAdam.optimizer_id = CPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.loss_scale = loss_scale
        try:
            import cpu_adam
        except ImportError:
            raise ImportError('Please install colossalai from source code to use CPUAdam')
        self.cpu_adam_op = cpu_adam
        self.cpu_adam_op.create_adam(self.opt_id,
                                     lr,
                                     betas[0],
                                     betas[1],
                                     eps,
                                     weight_decay,
                                     adamw_mode,
                                     simd_log)

    def __del__(self):
        self.cpu_adam_op.destroy_adam(self.opt_id)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                    "sure the cpu_offload is Ture"

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data,
                                                        dtype=torch.float,
                                                        device=device)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data,
                                                           dtype=torch.float,
                                                           device=device)
                    # memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                self.cpu_adam_op.adam_update(self.opt_id,
                                            state['step'],
                                            group['lr'],
                                            beta1,
                                            beta2,
                                            group['eps'],
                                            group['weight_decay'],
                                            group['bias_correction'],
                                            p.data,
                                            p.grad.data,
                                            state['exp_avg'],
                                            state['exp_avg_sq'],
                                            self.loss_scale)
        return loss

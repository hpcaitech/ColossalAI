# modified from https://github.com/NVIDIA/apex/blob/master/apex/optimizers/fused_sgd.py
import torch
from torch.optim.optimizer import Optimizer, required

from colossalai.registry import OPTIMIZERS
from colossalai.utils import multi_tensor_applier


@OPTIMIZERS.register_module
class FusedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    `FusedSGD` requires CUDA extensions which can be built during installation or runtime.

    This version of fused SGD implements 2 fusions.

      * Fusion of the SGD update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`colossalai.nn.optimizer.FusedSGD` may be used as a drop-in replacement for ``torch.optim.SGD``

    :class:`colossalai.nn.optimizer.FusedSGD` may be used with or without Amp.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 wd_after_momentum=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FusedSGD, self).__init__(params, defaults)

        self.wd_after_momentum = wd_after_momentum

        if multi_tensor_applier.available:
            from colossalai.kernel.op_builder import FusedOptimBuilder
            fused_optim = FusedOptimBuilder().load()

            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0],
                                                    dtype=torch.int,
                                                    device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_sgd = fused_optim.multi_tensor_sgd
        else:
            raise RuntimeError('FusedSGD requires cuda extensions')

    def __setstate__(self, state):
        super(FusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def get_momentums(self, params):
        momentums = []
        first_run = True
        for p in params:
            param_state = self.state[p]
            # torch.optim.SGD initializes momentum in the main loop, we have
            # to do it here, and track whether or not we've done so, so that
            # momentum application can be skipped in the main kernel.
            if 'momentum_buffer' not in param_state:
                first_run = True
                buf = param_state['momentum_buffer'] = torch.zeros_like(p)
                momentums.append(buf)
            else:
                first_run = False
                momentums.append(param_state['momentum_buffer'])
        return momentums, first_run

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # For each group, there are 3 possible combinations we need to consider:
            # grad_type, param_to_update_type, momentum_type
            # 1. fp16, fp16, fp16
            # 2. fp32, fp32, fp32
            # 3. fp16, fp32, fp32
            g_l, p_l = [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedSGD does not support sparse gradients')
                g_l.append(p.grad)
                p_l.append(p)
            m_l, first_run = self.get_momentums(p_l)
            multi_tensor_applier(self.multi_tensor_sgd, self._dummy_overflow_buf, [g_l, p_l, m_l], weight_decay,
                                 momentum, dampening, group['lr'], nesterov, first_run, self.wd_after_momentum, 1.0)

        return loss

# modified from https://github.com/NVIDIA/apex/blob/master/apex/optimizers/fused_sgd.py
import torch
from torch.optim.optimizer import Optimizer, required

from colossalai.registry import OPTIMIZERS
from ..multi_tensor_apply import multi_tensor_applier


@OPTIMIZERS.register_module
class FusedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Currently GPU-only.  Requires ColossalAI to be installed via
    ``pip install -v --no-cache-dir --global-option="--cuda_ext" ./``.

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

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 wd_after_momentum=False,
                 materialize_master_grads=True,
                 set_grad_none=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(FusedSGD, self).__init__(params, defaults)

        self.wd_after_momentum = wd_after_momentum
        self.materialize_master_grads = materialize_master_grads
        self.most_recent_scale = 1.0
        self.scale_set_by_backward = False
        self.set_grad_none = set_grad_none

        if multi_tensor_applier.available:
            import colossal_C
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor(
                [0], dtype=torch.int, device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_sgd = colossal_C.multi_tensor_sgd
        else:
            raise RuntimeError(
                'apex.optimizers.FusedSGD requires cuda extensions')

    def __setstate__(self, state):
        super(FusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedSGD, self).zero_grad()

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
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
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

        explicit_master_params = (hasattr(self, "_amp_stash") and
                                  hasattr(self._amp_stash, "fp32_from_fp16_groups"))

        for gid, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # For each group, there are 3 possible combinations we need to consider:
            # grad_type, param_to_update_type, momentum_type, requires_fp16_model_copy
            # 1. fp16, fp16, fp16, No
            # 2. fp32, fp32, fp32, No
            # 3. fp16, fp32, fp32, Yes

            first_runs = [True, True]

            # I think a bit of code divergence in exchange for naming clarity is worthwhile
            if explicit_master_params:
                stash = self._amp_stash

                fp32_params = [
                    p for p in stash.fp32_from_fp32_groups[gid] if p.grad is not None]
                fp32_grads = [
                    p.grad for p in stash.fp32_from_fp32_groups[gid] if p.grad is not None]
                fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)

                if self.materialize_master_grads:
                    fp16_model_params = [p for i, p in enumerate(
                        stash.fp16_groups[gid]) if stash.fp32_from_fp16_groups[gid][i].grad is not None]
                    fp32_from_fp16_grads = [
                        p.grad for p in stash.fp32_from_fp16_groups[gid] if p.grad is not None]
                    fp32_from_fp16_params = [
                        p for p in stash.fp32_from_fp16_groups[gid] if p.grad is not None]
                    fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(
                        fp32_from_fp16_params)

                    fp16_set = [fp32_from_fp16_grads, fp32_from_fp16_params,
                                fp32_from_fp16_momentums, fp16_model_params]
                else:
                    fp16_model_params = [
                        p for p in stash.fp16_groups[gid] if p.grad is not None]
                    fp16_model_grads = [
                        p.grad for p in stash.fp16_groups[gid] if p.grad is not None]
                    fp32_from_fp16_params = [p for i, p in enumerate(
                        stash.fp32_from_fp16_groups[gid]) if stash.fp16_groups[gid][i].grad is not None]
                    fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(
                        fp32_from_fp16_params)

                    fp16_set = [fp16_model_grads, fp32_from_fp16_params,
                                fp32_from_fp16_momentums, fp16_model_params]

                launch_sets = [fp16_set, [
                    fp32_grads, fp32_params, fp32_momentums]]
            else:
                fp16_params = [p for p in group['params'] if (
                    p.dtype == torch.float16 and p.grad is not None)]
                fp16_grads = [p.grad for p in group['params'] if (
                    p.dtype == torch.float16 and p.grad is not None)]
                fp16_momentums, first_runs[0] = self.get_momentums(fp16_params)

                fp32_params = [p for p in group['params'] if (
                    p.dtype == torch.float32 and p.grad is not None)]
                fp32_grads = [p.grad for p in group['params'] if (
                    p.dtype == torch.float32 and p.grad is not None)]
                fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)

                launch_sets = [[fp16_grads, fp16_params, fp16_momentums],
                               [fp32_grads, fp32_params, fp32_momentums]]

            for s, (launch_set, first_run) in enumerate(zip(launch_sets, first_runs)):
                assert len(launch_set[0]) == len(launch_set[1])
                assert len(launch_set[0]) == len(launch_set[2])
                if len(launch_set[0]) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_sgd,
                        self._dummy_overflow_buf,
                        launch_set,
                        weight_decay,
                        momentum,
                        dampening,
                        group['lr'],
                        nesterov,
                        first_run,
                        self.wd_after_momentum,
                        1.0 / self.most_recent_scale)

        self.most_recent_scale = 1.0
        self.scale_set_by_backward = False

        return loss

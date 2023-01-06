#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.optim import Optimizer

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.kernel.op_builder import FusedOptimBuilder
from colossalai.logging import get_dist_logger
from colossalai.utils import clip_grad_norm_fp32, copy_tensor_parallel_attributes, multi_tensor_applier

from ._utils import has_inf_or_nan, zero_gard_by_list
from .grad_scaler import BaseGradScaler

try:
    from colossalai._C import fused_optim
except:
    fused_optim = None

__all__ = ['FP16Optimizer']


def load_fused_optim():
    global fused_optim

    if fused_optim is None:
        fused_optim = FusedOptimBuilder().load()


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """
    adapted from Megatron-LM (https://github.com/NVIDIA/Megatron-LM)

    Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        global fused_optim
        load_fused_optim()
        multi_tensor_applier(fused_optim.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class FP16Optimizer(Optimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD
        grad_scaler (BaseGradScaler): grad scaler for gradient chose in
                                      ``constant_grad_scaler`` or ``dynamic_grad_scaler``.
        clip_grad_norm (float, optional): clip gradients with this global L2 norm. Default 0.
                        Note that clipping is ignored if clip_grad == 0
        verbose (bool, optional): if set to `True`, will print debug info. Default False.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 grad_scaler: BaseGradScaler,
                 verbose: bool = False,
                 clip_grad_norm=0,
                 dp_process_group: ProcessGroup = None,
                 mp_process_group: ProcessGroup = None):
        # have a defaults for compatibility with pytorch optim
        self._optimizer = optimizer
        self._defaults = optimizer.defaults

        # fp16-related params
        assert isinstance(grad_scaler, BaseGradScaler)
        self._grad_scaler = grad_scaler
        self._found_overflow = torch.cuda.FloatTensor([0.0])
        self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # misc params
        self._clip_grad_max_norm = clip_grad_norm

        # get process group
        def _get_process_group(parallel_mode):
            if gpc.is_initialized(parallel_mode) and gpc.get_world_size(parallel_mode):
                return gpc.get_group(parallel_mode)
            else:
                return None

        if dp_process_group is None:
            dp_process_group = _get_process_group(ParallelMode.DATA)
        if mp_process_group is None:
            mp_process_group = _get_process_group(ParallelMode.MODEL)

        self._dp_process_group = dp_process_group
        self._mp_process_group = mp_process_group

        # we maintain three groups of parameters
        # so that the model can have a mixture
        # of fp16 and fp32 params
        # fp16_param_groups: the fp16 params of the model
        # fp32_master_param_groups: the fp32 params cast from the fp16 param of the model
        # fp32_param_groups: the fp32 params of the model
        # NOTE:
        # 1. fp16_param_groups and fp32_master_param_groups have one-to-one correspondence
        # 2. fp32_param_groups and fp16_param_groups are exclusive of each other
        self._fp16_param_groups = []
        self._fp32_master_param_groups = []
        self._fp32_param_groups = []

        # For all the groups in the original optimizer:
        for param_group in self._optimizer.param_groups:
            fp16_params = []
            fp32_master_params = []
            fp32_params = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor']:
                        fp16_params.append(param)

                        # Create a fp32 copy
                        fp32_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        copy_tensor_parallel_attributes(param, fp32_param)

                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = fp32_param
                        fp32_master_params.append(fp32_param)

                        # Reset existing state dict key to the new main param.
                        if param in self._optimizer.state:
                            self._optimizer.state[fp32_param] = self._optimizer.state.pop(param)

                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params.append(param)
                    else:
                        raise TypeError('Expected parameter of type torch.cuda.FloatTensor '
                                        f'or torch.cuda.HalfTensor, but got {param.type()}')

            self._fp16_param_groups.append(fp16_params)
            self._fp32_master_param_groups.append(fp32_master_params)
            self._fp32_param_groups.append(fp32_params)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self._optimizer.load_state_dict(self._optimizer.state_dict())

        # log config
        self._logger = get_dist_logger()
        if verbose:
            self._logger.info(
                f"\n=========  FP16 Optimizer Config =========\n"
                f"Optimizer: {optimizer.__class__.__name__}\n"
                f"clip_grad_norm = {clip_grad_norm}\n"
                f"grad_scaler = {self._grad_scaler.__class__.__name__}"
                f"==========================================",
                ranks=[0])

    @property
    def max_norm(self):
        """Returns the maximum norm of gradient clipping.
        """
        return self._clip_grad_max_norm

    @property
    def grad_scaler(self):
        """Returns the gradient scaler.

        Returns:
            :class:`BaseGradScaler`: gradient scaler.
        """

        return self._grad_scaler

    @property
    def loss_scale(self):
        """Returns the loss scale.

        Returns:
            int: loss scale.
        """
        return self._grad_scaler.scale

    @property
    def optimizer(self):
        """Returns the optimizer.

        Returns:
            :class:`torch.optim.Optimizer`: the optimizer object wrapped.
        """
        return self._optimizer

    @property
    def defaults(self):
        """Returns the default arguments of optimizer.

        Returns:
            dict: optimizer arguments saved in defaults of the optimizer wrapped.
        """
        return self._defaults

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and has_inf_or_nan(p.grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce across dp group
        if self._dp_process_group:
            dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self._dp_process_group)

        # all-reduce over model parallel group
        if self._mp_process_group:
            dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self._mp_process_group)

        return self._found_overflow.item() > 0

    def zero_grad(self, set_to_none=True):
        """Set gradient to zero.

        Args:
            set_to_none (bool): Whether set the gradient to None.
        """

        # set_to_none = True can save some memory space
        for param_group in self._optimizer.param_groups:
            zero_gard_by_list(param_group['params'], set_to_none=set_to_none)

    def _get_fp32_param_groups_to_update(self):
        return self._fp32_master_param_groups + self._fp32_param_groups

    def _unscale_grads(self):
        for group in self._get_fp32_param_groups_to_update():
            for p in group:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)

    def _assign_grad_to_fp32_master_param(self):
        # This only needs to be done for the float16 group.
        for fp16_param_group, fp32_master_param_group in zip(self._fp16_param_groups, self._fp32_master_param_groups):
            for fp16_param, fp32_param in zip(fp16_param_group, fp32_master_param_group):
                if fp16_param.grad is not None:
                    fp32_param.grad = fp16_param.grad.float()
                    # clear unneeded grad on fp16 param
                    fp16_param.grad = None

    def _update_fp16_param_from_fp32_param(self):
        fp16_param_data = []
        fp32_master_param_data = []
        for fp16_group, fp32_group in zip(self._fp16_param_groups, self._fp32_master_param_groups):
            for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                fp16_param_data.append(fp16_param.data)
                fp32_master_param_data.append(fp32_param.data)
        _multi_tensor_copy_this_to_that(this=fp32_master_param_data,
                                        that=fp16_param_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def step(self):
        """Update the model parameters.
        """

        # Copy gradients from model params to main params.
        self._assign_grad_to_fp32_master_param()
        self._unscale_grads()

        overflow = self._check_overflow()
        self._grad_scaler.update(overflow)
        if overflow:
            self.zero_grad()

        # Clip the main gradients.
        grad_norm = None
        if self._clip_grad_max_norm > 0.0:
            grad_norm = self.clip_grad_norm(self._clip_grad_max_norm)

        if not overflow:
            # Step the optimizer.
            self._optimizer.step()

            # Update params from main params.
            self._update_fp16_param_from_fp32_param()

            # Successful update.
            return True, grad_norm
        else:
            return False, None

    def backward(self, loss):
        """Execute backward pass.

        Args:
            loss (:class:`torch.Tensor`): the loss value.
        """

        scaled_loss = loss * self.grad_scaler.scale
        scaled_loss.backward()

    def state_dict(self):
        """Returns the states of the fp16 optimizer as a dict object.
        """

        state_dict = {}
        state_dict['optimizer'] = self._optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_master_param_groups'] = self._fp32_master_param_groups
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the states of the fp16 optimizer from a dict object.

        Args:
            state_dict (dict): the states of the fp16 optimizer
        """

        # Optimizer.
        self._optimizer.load_state_dict(state_dict['optimizer'])

        # Grad scaler.
        if 'grad_scaler' in state_dict:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])

        # Copy data for the main params.
        if 'fp32_master_param_groups' in state_dict:
            for current_group, ckpt_group in zip(self._fp32_master_param_groups,
                                                 state_dict['fp32_master_param_groups']):
                for current_param, ckpt_param in zip(current_group, ckpt_group):
                    current_param.data.copy_(ckpt_param.data)

    def clip_grad_norm(self, clip_grad):
        """Clip gradients by norm.

        Args:
            clip_grad (float): the max norm for clipping
        """
        params = []
        for param_group in self._optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return clip_grad_norm_fp32(params, clip_grad)

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self._optimizer.state

    def _set_state(self, value):
        self._optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self._optimizer.param_groups

    def _set_param_groups(self, value):
        self._optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

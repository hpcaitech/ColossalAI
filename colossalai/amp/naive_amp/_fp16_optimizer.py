#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

try:
    import colossal_C
except:
    print('Colossalai should be built with cuda extension to use the FP16 optimizer')

from torch.optim import Optimizer

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import (print_rank_0, copy_tensor_parallel_attributes,
                              clip_grad_norm_fp32, count_zeros_fp32, multi_tensor_applier, is_using_pp)


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(colossal_C.multi_tensor_scale,
                             overflow_buf,
                             [this, that],
                             1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class DynamicGradScaler:

    def __init__(self,
                 initial_scale,
                 min_scale,
                 growth_factor,
                 backoff_factor,
                 growth_interval,
                 hysteresis,
                 max_scale: int = None,
                 verbose: bool = False):
        """"Grad scaler with dynamic scale that gets adjusted
        during training."""
        assert initial_scale > 0.0
        self._scale = torch.cuda.FloatTensor([initial_scale])

        # Lower bound on the scale.
        assert min_scale > 0.0
        assert min_scale <= initial_scale
        self.min_scale = torch.cuda.FloatTensor([min_scale])
        # Growth and backoff factors for the scale.
        assert growth_factor > 1.0
        self.growth_factor = torch.cuda.FloatTensor([growth_factor])
        assert backoff_factor < 1.0
        assert backoff_factor > 0.0
        self.backoff_factor = torch.cuda.FloatTensor([backoff_factor])
        # Interval over which if we don't see any inf/nan,
        # we will scale the grad scale by the growth factor.
        assert growth_interval > 0
        self.growth_interval = growth_interval
        # Number of inf/nans we should see before scaling down
        # the grad scale by the backoff factor.
        assert hysteresis > 0
        self.hysteresis = hysteresis
        if max_scale is not None:
            assert max_scale > 1 and initial_scale <= max_scale
        self._max_scale = max_scale

        # Trackers.
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis

        self._logger = get_dist_logger()
        self.verbose = verbose

    @property
    def scale(self):
        return self._scale

    @property
    def inv_scale(self):
        return self._scale.double().reciprocal().float()

    def update(self, found_inf):

        # If we have an inf/nan, growth tracker is set to 0
        # and hysterisis tracker is reduced by 1.
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            # Now if we are out of hysteresis count, scale down the loss.
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor,
                                        self.min_scale)
            if self.verbose:
                self._logger.info(f'overflow occurs, loss scale is adjusted to {self._scale}', ranks=[0])
        else:
            # If there is no nan/inf, increment the growth tracker.
            self._growth_tracker += 1
            # If we have had enough consequitive intervals with no nan/inf:
            if self._growth_tracker == self.growth_interval:
                # Reset the tracker and hysteresis trackers,
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                # and scale up the loss scale.
                if self._max_scale is not None and self._scale >= self._max_scale:
                    if self.verbose:
                        self._logger.info(
                            f'Current loss scale {self._scale} has reached the max scale {self._max_scale} allowed', ranks=[0])
                else:
                    self._scale = self._scale * self.growth_factor
                    if self.verbose:
                        self._logger.info(
                            f'no consecutive overflow, loss scale is adjusted to {self._scale}', ranks=[0])

    def state_dict(self):
        state_dict = {}
        state_dict['max_scale'] = self._max_scale
        state_dict['scale'] = self._scale
        state_dict['growth_tracker'] = self._growth_tracker
        state_dict['hysteresis_tracker'] = self._hysteresis_tracker
        return state_dict

    def load_state_dict(self, state_dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']
        self._max_scale = state_dict['max_scale']


class FP16Optimizer(Optimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    :param optimizer: base optimizer such as Adam or SGD
    :type optimizer: torch.optim.Optimizer
    :param clip_grad: clip gradeints with this global L2 norm. Note that clipping is ignored if clip_grad == 0
    :type param clip_grad: float
    :param log_num_zeros_in_grad: return number of zeros in the gradients.
    :type log_num_zeros_in_grad: bool
    :param initial_scale: initial scale of gradient scaler
    :type initial_scale: int
    :param growth_factor: the growth rate of loss scale
    :type growth_factor: int
    :param backoff_factor: the decrease rate of loss scale
    :type backoff_factor: float
    :param hysterisis: delay shift in dynamic loss scaling
    :type hysterisis: int
    :param max_scale: maximum loss scale allowed
    :type max_scale: int
    :param verbose: if set to `True`, will print debug info
    :type verbose: bool
    """

    def __init__(self,
                 optimizer,
                 clip_grad=0,
                 log_num_zeros_in_grad=False,
                 initial_scale=2 ** 32,
                 min_scale=1,
                 growth_factor=2,
                 backoff_factor=0.5,
                 growth_interval=1000,
                 hysteresis=2,
                 max_scale: int = 2 ** 32,
                 verbose: bool = False):
        # default args for compatibility
        bf16 = False
        params_have_main_grad = False

        # have a defaults for compatibility with pytorch optim
        self.defaults = optimizer.defaults

        # log config
        self._logger = get_dist_logger()
        if verbose:
            self._logger.info(f"\n=========  FP16 Optimizer Config =========\n"
                              f"Optimizer: {optimizer.__class__.__name__}\n"
                              f"clip_grad = {clip_grad}\n"
                              f"log_num_zeros_in_grad = {log_num_zeros_in_grad}\n"
                              f"initial_scale = {initial_scale}\n"
                              f"min_scale = {min_scale}\n"
                              f"growth_factor = {growth_factor}\n"
                              f"backoff_factor = {backoff_factor}\n"
                              f"growth_interval = {growth_interval}\n"
                              f"hysteresis = {hysteresis}\n"
                              f"==========================================", ranks=[0])

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad

        self.bf16 = bf16
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            verbose=verbose
        )

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])

        # ======================
        # main parameter stuff
        # ======================

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor',
                                        'torch.cuda.BFloat16Tensor']:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        copy_tensor_parallel_attributes(param, main_param)

                        # if hasattr(param, 'shared'):
                        #     main_param.shared = param.shared

                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] \
                                = self.optimizer.state.pop(param)

                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param
                    else:
                        raise TypeError('Wrapped parameters must be one of '
                                        'torch.cuda.FloatTensor,  '
                                        'torch.cuda.HalfTensor, or '
                                        'torch.cuda.BFloat16Tensor. '
                                        'Received {}'.format(param.type()))

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(
                fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def zero_grad(self, set_to_none=False):
        """We only need to zero the model related parameters, i.e.,
                float16_groups & fp32_from_fp32_groups."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if self.params_have_main_grad:
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

    def _unscale_main_grads_and_check_for_nan(self):
        main_grads = []
        # fp32 params fromm float16 ones.
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Append fp32 parameters.
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=gpc.get_group(ParallelMode.MODEL))

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)
        return found_inf_flag

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups,
                                           self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self):
        # Copy gradients from model params to main params.
        self._copy_model_grads_to_main_grads()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if \
            self.log_num_zeros_in_grad else None

        # Step the optimizer.
        self.optimizer.step()

        # Update params from main params.
        self._copy_main_params_to_model_params()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        # Optimizer.
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from '
                         'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            print_rank_0('***WARNING*** found an old checkpoint, will not '
                         'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')

        # Copy data for the main params.
        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(
                self.fp32_from_float16_groups,
                state_dict[fp32_from_float16_params_key]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        return clip_grad_norm_fp32(params, clip_grad)

    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params)

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

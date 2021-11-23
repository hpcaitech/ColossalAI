#!/usr/bin/env python
# -*- encoding: utf-8 -*-

try:
    import apex.amp as apex_amp
except:
    pass

try:
    import torch.cuda.amp as torch_amp
except:
    pass

from typing import Iterable

import torch.nn as nn
from torch.optim import Optimizer

from colossalai.nn import (ZeroRedundancyOptimizer_Level_2,
                           ZeroRedundancyOptimizer_Level_3)
from colossalai.nn.optimizer._utils import clip_grad_norm_fp32
from ._base_schedule import BaseSchedule
from ._utils import convert_to_fp16, convert_to_fp32
from ..amp import AMP_TYPE, GradScaler


class NoPipelineSchedule(BaseSchedule):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    :param amp_type: The type of automatic mixed precision
    :param amp_config: The configuration of automatic mixed procision
    :type amp_type: AMP_TYPE
    :type amp_config: dict
    """

    def __init__(
            self,
            amp_type: AMP_TYPE = None,
            amp_config: dict = None,
    ):
        super().__init__()

        # mixed precision training
        assert amp_type is None or isinstance(amp_type, AMP_TYPE), \
            'unrecognised value for argument fp16, it can only be None, torch or apex'

        self.use_zero_level_2_3 = False

        if amp_type is not None:
            self.fp16 = True
            self.amp_type = amp_type

            if amp_config is not None:
                assert isinstance(amp_config, dict), \
                    f'expected argument fp16_config to be type dictionary, but got {type(amp_config)}'

            if self.amp_type == AMP_TYPE.TORCH:
                # torch apex
                if amp_config is None:
                    amp_config = dict()
                self.amp_cfg = amp_config
            elif self.amp_type == AMP_TYPE.APEX:
                # apex amp
                if amp_config is None:
                    amp_config = dict(opt_level='O2')
                self.logger.warning(
                    'apex is deprecated, please consider using torch.cuda.amp instead.'
                )
                self.amp_cfg = amp_config
            elif self.amp_type == AMP_TYPE.PARALLEL:
                # use fp16 optimizer for tensor parallel training
                if amp_config is None:
                    amp_config = dict()
                self.amp_cfg = amp_config
        else:
            self.fp16 = False
            self.amp_type = None

    def initialize(self, model: nn.Module, optimizer: Optimizer):
        if isinstance(optimizer, (ZeroRedundancyOptimizer_Level_2,
                                  ZeroRedundancyOptimizer_Level_3)):
            self.use_zero_level_2_3 = True
            assert self.amp_type != AMP_TYPE.PARALLEL, \
                'ZeRO Level 2 and 3 are mutually exclusive with AMP_TYPE.PARALLEL'

        if self.fp16:
            if self.amp_type == AMP_TYPE.TORCH:
                self._torch_amp_scaler = GradScaler(**self.amp_cfg)
            elif self.amp_type == AMP_TYPE.APEX:
                model, optimizer = apex_amp.initialize(model, optimizer, **self.amp_cfg)

        return model, optimizer

    def forward_backward_step(self,
                              data_iter: Iterable,
                              model: nn.Module,
                              criterion: nn.modules.loss._Loss,
                              optimizer: Optimizer = None,
                              forward_only: bool = False,
                              grad_accum_size: int = 1,
                              return_loss: bool = True):
        """The process function that loads loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        :param data_iter: Data iterator of the dataloader, e.g. iter(dataloader)
        :param model: Model for training and inference
        :param criterion: Loss function for training
        :param optimizer: Optimizer used for training
        :param forward_only: If True, the model is run for the forward pass, else back propagation will be executed
        :param grad_accum_size: The number of iterations for gradient accumulation
        :param return_loss: Loss will be returned if True
        :type data_iter: Iterator
        :type model: torch.nn.Module
        :type criterion: torch.nn.modules.loss._Loss
        :type optimizer: torch.optim.Optimizer
        :type forward_only: bool, optional
        :type grad_accum_size: int
        :type return_loss: bool, optional
        :return: (output, label, loss)
        """
        assert forward_only or return_loss, \
            'The argument \'return_loss\' has to be True when \'forward_only\' is False, but got False.'

        data, label = self.load_batch(data_iter)
        loss = None

        # forward
        if self.fp16 and self.amp_type == AMP_TYPE.TORCH:
            with torch_amp.autocast():
                output = model(*data)
                if not isinstance(output, (tuple, list)):
                    output = (output,)
                if return_loss:
                    loss = criterion(*output, *label)
        else:
            if self.use_zero_level_2_3 or self.amp_type == AMP_TYPE.PARALLEL:
                data = convert_to_fp16(data)

            output = model(*data)

            if self.use_zero_level_2_3 or self.amp_type == AMP_TYPE.PARALLEL:
                output = convert_to_fp32(output)

            if not isinstance(output, (tuple, list)):
                output = (output,)
            if return_loss:
                loss = criterion(*output, *label)

        loss /= grad_accum_size

        if not forward_only:
            # backward
            if self.use_zero_level_2_3:
                optimizer.backward(loss)
            elif self.fp16:
                if self.amp_type == AMP_TYPE.APEX:
                    with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif self.amp_type == AMP_TYPE.TORCH:
                    self._torch_amp_scaler.scale(loss).backward()
                elif self.amp_type == AMP_TYPE.PARALLEL:
                    loss = optimizer.scale_loss(loss)
                    loss.backward()
                    # scale back to display the original value in logs
                    loss.div_(optimizer.grad_scaler.scale)
            else:
                loss.backward()

        if return_loss:
            return output, label, loss * grad_accum_size
        else:
            return output, None, None

    def optimizer_step(self, model: nn.Module, optimizer: Optimizer, grad_clipping: float = 0.0):
        # step optimizer
        if self.fp16 and self.amp_type == AMP_TYPE.TORCH:
            if grad_clipping > 0.0:
                self._torch_amp_scaler.unscale_(optimizer)
                clip_grad_norm_fp32(model.parameters(), grad_clipping)
            self._torch_amp_scaler.step(optimizer)
            self._torch_amp_scaler.update()
        else:
            if not self.fp16 and not self.use_zero_level_2_3 and grad_clipping > 0.0:
                clip_grad_norm_fp32(model.parameters(), grad_clipping)
            optimizer.step()

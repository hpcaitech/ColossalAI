#!/usr/bin/env python
# -*- encoding: utf-8 -*-

try:
    import apex.amp as apex_amp
except:
    print('apex is required for mixed precision training')
try:
    import torch.cuda.amp as torch_amp
except:
    print('PyTorch amp is not supported with the current PyTorch version')

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.amp_type import AMP_TYPE
from colossalai.nn import (ZeroRedundancyOptimizer_Level_2,
                           ZeroRedundancyOptimizer_Level_3)
from ._utils import convert_to_fp16
from ._base_schedule import BaseSchedule


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

        # LSG: check compatibility
        # LSG: torch.cuda.amp and apex.amp cannot be used for tensor parallel
        if gpc.is_initialized(ParallelMode.TENSOR) and gpc.get_world_size(
                ParallelMode.TENSOR) > 1:
            assert amp_type != AMP_TYPE.TORCH and amp_type != AMP_TYPE.APEX, \
                'You can only AMP_TYPE.PARALLEL for tensor parallel training'
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

    @property
    def num_steps(self):
        return len(self.dataloader)

    def initialize(self,
                   dataloader,
                   model,
                   criterion,
                   optimizer,
                   lr_scheduler=None):
        super().initialize(dataloader,
                           model,
                           criterion,
                           optimizer,
                           lr_scheduler=lr_scheduler)
        if isinstance(self.optimizer, (ZeroRedundancyOptimizer_Level_2,
                                       ZeroRedundancyOptimizer_Level_3)):
            self.use_zero_level_2_3 = True
            assert self.amp_type != AMP_TYPE.PARALLEL, 'ZeRO Level 2 and 3 are mutually exclusive with AMP_TYPE.PARALLEL'

        if self.fp16:
            if self.amp_type == AMP_TYPE.TORCH:
                self._torch_amp_scaler = torch_amp.GradScaler(**self.amp_cfg)
            elif self.amp_type == AMP_TYPE.APEX:
                self.model, self.optimizer = apex_amp.initialize(
                    self.model, self.optimizer, **self.amp_cfg)

    def forward_backward_step(self, forward_only=False, return_loss=True):
        """The process function that loads loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        :return: (output, label, loss)
        """
        assert forward_only or return_loss, \
            'The argument \'return_loss\' has to be True when \'forward_only\' is False, but got False.'

        data, label = self.load_batch()
        loss = None

        # LSG: leave for debug, make sure dataloader is deterministic
        # if forward_only:
        #     img = data[0]
        #     rank = gpc.get_local_rank(ParallelMode.DATA)
        #     world_size = gpc.get_world_size(ParallelMode.DATA)
        #     group = gpc.get_group(ParallelMode.DATA)
        #     input_list = [img.clone() for _ in range(world_size)]
        #     output_list = [torch.empty_like(img) for _ in range(world_size)]
        #     output_list[rank] = img.clone()
        #     dist.all_to_all(output_tensor_list=output_list, input_tensor_list=input_list, group=group)
        #     assert torch.equal(output_list[0], output_list[1])  # and torch.equal(output_list[1], output_list[2])

        # forward
        if self.fp16 and self.amp_type == AMP_TYPE.TORCH:
            with torch_amp.autocast():
                output = self.model(*data)
                if not isinstance(output, (tuple, list)):
                    output = (output,)
                if return_loss:
                    loss = self.criterion(*output, *label)
        else:
            if self.use_zero_level_2_3 or self.amp_type == AMP_TYPE.PARALLEL:
                data = convert_to_fp16(data)

            output = self.model(*data)
            if not isinstance(output, (tuple, list)):
                output = (output,)
            if return_loss:
                loss = self.criterion(*output, *label)

        if not forward_only:
            # backward
            if self.use_zero_level_2_3:
                self.optimizer.backward(loss)
            elif self.fp16:
                if self.amp_type == AMP_TYPE.APEX:
                    with apex_amp.scale_loss(loss,
                                             self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif self.amp_type == AMP_TYPE.TORCH:
                    self._torch_amp_scaler.scale(loss).backward()
                elif self.amp_type == AMP_TYPE.PARALLEL:
                    loss = self.optimizer.scale_loss(loss)
                    loss.backward()
                    # scale back to display the original value in logs
                    loss.div_(self.optimizer.grad_scaler.scale)
            else:
                loss.backward()

        if return_loss:
            return output, label, loss
        else:
            return output, None, None

    def step(self):
        # step optimizer
        if self.fp16 and self.amp_type == AMP_TYPE.TORCH:
            self._torch_amp_scaler.step(self.optimizer)
            self._torch_amp_scaler.update()
        else:
            self.optimizer.step()

        # update lr scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

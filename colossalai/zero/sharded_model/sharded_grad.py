from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ShardedGradient:
    def __init__(self,
                 param: Parameter,
                 sharded_module: nn.Module,
                 offload_config: Optional[dict] = None
                 ) -> None:
        assert hasattr(
            param, 'ca_attr') and param.ca_attr.is_sharded, 'ShardedGradient can only be initialized with sharded parameter'

        self.param = param
        self.sharded_module = sharded_module
        self.offload_config = offload_config

        self._cpu_offload = offload_config.get('device', None) == 'cpu' if offload_config else False

        # _gpu_grad is either sharded or not
        # all saved grads are fp32
        self._gpu_grad: Optional[torch.Tensor] = None
        self._cpu_grad: Optional[torch.Tensor] = None

        if self._cpu_offload:
            # this buffer will be held and reused every iteration
            self._cpu_grad = torch.zeros(param.ca_attr.payload('cpu'), dtype=torch.float).pin_memory()

    @torch.no_grad()
    def setup(self) -> None:
        """This function will be called pre-backward. Save the local accumulated gradient to _gpu_grad.
        When no_sync() is enable (_require_backward_grad_sync=False), the grad is accumulated locally in param.grad

        :raises AssertionError: Raise if grad shape is wrong
        """
        if self.sharded_module._require_backward_grad_sync and self.param.grad is not None:
            if self.param.grad.device != self.param.data.device:
                # TODO: offload?
                raise RuntimeError(
                    'grad and param are on different device, grad {self.param.grad.device} vs. param {self.param.data.device}')
            else:
                self._gpu_grad = self.param.grad.data
            self.param.grad = None

    def reduce_scatter_callback(self, reduced_grad: torch.Tensor) -> None:
        """This function will be called in post-backward hook, so we cannot modify param.grad directly

        :param reduced_grad: the reduced grad
        :type reduced_grad: torch.Tensor
        """
        # Make sure we store fp32 grad
        if torch.is_floating_point(reduced_grad) and reduced_grad.dtype != torch.float:
            reduced_grad.data = reduced_grad.data.to(torch.float)

        if self._gpu_grad is None:
            self._gpu_grad = reduced_grad.data
        else:
            self._gpu_grad += reduced_grad.data

        # Optionally move gradients to CPU, typically used if one is running the optimizer on the CPU. Once the full
        # backwards pass completes, we will set `.grad` to the CPU copy.
        if self._cpu_offload:
            self._cpu_grad.copy_(self._gpu_grad.data, non_blocking=True)
            # Don't let this memory get reused until after the transfer.
            self._gpu_grad.data.record_stream(torch.cuda.current_stream())

    @torch.no_grad()
    def write_back(self) -> None:
        """This function will be called in final backward hook
        """
        if self._cpu_grad is not None:
            assert self.param.device == torch.device(
                'cpu'), f'Incorrect param device, expected CPU, got {self.param.device}'
            self.param.grad.data = self._cpu_grad
        elif self._gpu_grad is not None:
            assert self.param.device == self._gpu_grad.device, f'Incorrect _gpu_grad device, param on {self.param.device} but _gpu_grad on {self._gpu_grad.device}'
            self.param.grad.data = self._gpu_grad
        else:
            raise RuntimeError('No grad to write back')
        # If using CPU offload, _cpu_grad will store the CPU tensor of _gpu_grad
        # They should be released here
        self._gpu_grad = None

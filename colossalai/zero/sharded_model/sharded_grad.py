from typing import Optional

import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter


class ShardedGradient:
    def __init__(self,
                 param: Parameter,
                 process_group: Optional[ProcessGroup] = None,
                 offload_config: Optional[dict] = None
                 ) -> None:
        assert hasattr(
            param, 'ca_attr') and param.ca_attr.is_shared, 'ShardedGradient can only be initialized with sharded parameter'

        self.param = param
        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.shard_idx = self.process_group.rank()
        self.num_shards = self.process_group.size()
        self.offload_config = offload_config

        self._cpu_offload = offload_config.get('device', None) == 'cpu' if offload_config else False

        self._is_sharded = self.num_shards > 1
        self._orig_size = param.ca_attr.origin_shape
        self._shard_size = param.ca_attr.shard_shape

        self._saved_grad_shard: Optional[torch.Tensor] = None
        self._saved_full_grad: Optional[torch.Tensor] = None
        self._cpu_grad: Optional[torch.Tensor] = None

        if self._cpu_offload:
            # this buffer will be held and reused every iteration
            self._cpu_grad = torch.zeros_like(param.ca_attr.payload('cpu')).pin_memory()

    @torch.no_grad()
    def setup(self) -> None:
        """This function will be called pre-backward. Save the local accumulated gradient to _saved_grad_shard or _saved_full_grad

        :raises AssertionError: Raise if grad shape is wrong
        """
        if self.param.grad is not None:
            set_grad_to_none = True
            if self.param.grad.device != self.param.data.device:
                # TODO: offload?
                raise RuntimeError(
                    'grad and param are on different device, grad {self.param.grad.device} vs. param {self.param.data.device}')
            elif self.param.grad.size() == self._orig_size:
                if not self._is_sharded:
                    self._saved_full_grad = self.param.grad.data
                else:
                    # This is gradient accumulation with no_sync context.
                    set_grad_to_none = False
            elif self.param.grad.size() == self._shard_size:
                # This is gradient accumulation without no_sync context.
                # We save the grad shard and set p.grad to None for this backward pass.
                # We will accumulate after this pass's grad is generated and reduced and
                # sharded.
                self._saved_grad_shard = self.param.grad.data
            else:
                raise AssertionError(f"unexpected grad shape: {self.param.grad.size()}")
            if set_grad_to_none:
                self.param.grad = None

    def reduce_scatter_callback(self, reduced_grad: torch.Tensor) -> None:
        """This function will be called in post-backward hook, so we cannot modify param.grad directly

        :param reduced_grad: the reduced grad
        :type reduced_grad: torch.Tensor
        """
        if self._is_sharded:
            # Accumulate into the gradient shard.
            if self._saved_grad_shard is None:
                self._saved_grad_shard = reduced_grad.data
            else:
                assert (self._saved_grad_shard.shape == reduced_grad.shape
                        ), f'{self._saved_grad_shard.shape} vs {reduced_grad.shape}'
                self._saved_grad_shard.data += reduced_grad.data
            reduced_grad = self._saved_grad_shard.data
        else:
            # We can't modify the dtype of grad in this function
            # So we use `_saved_full_grad` to store gradient
            # This is useful when using mixed precision mode on single node
            if self._saved_full_grad is None:
                self._saved_full_grad = reduced_grad.data
            else:
                self._saved_full_grad += reduced_grad.data

        # Optionally move gradients to CPU, typically used if one is running the optimizer on the CPU. Once the full
        # backwards pass completes, we will set `.grad` to the CPU copy.
        if self._cpu_offload:
            self._cpu_grad.copy_(reduced_grad.data, non_blocking=True)
            # Don't let this memory get reused until after the transfer.
            reduced_grad.data.record_stream(torch.cuda.current_stream())

    @torch.no_grad()
    def write_back(self) -> None:
        """This function will be called in final backward hook
        """
        if self._cpu_grad is not None:
            assert self.param.device == torch.device(
                'cpu'), f'Incorrect param device, expected CPU, got {self.param.device}'
            self.param.grad.data = self._cpu_grad
        elif self._saved_grad_shard is not None:
            assert self.param.device == self._saved_grad_shard.device, f'Incorrect _saved_grad_shard device, param on {self.param.device} but _saved_grad_shard on {self._saved_grad_shard.device}'
            self.param.grad.data = self._saved_grad_shard
        elif self._saved_full_grad is not None:
            self.param.grad.data = self._saved_full_grad
        else:
            raise RuntimeError('No grad to write back')
        # If using CPU offload, _cpu_grad will store the CPU tensor of _saved_grad_shard or _saved_full_grad
        # They should be released here
        self._saved_grad_shard = None
        self._saved_full_grad = None

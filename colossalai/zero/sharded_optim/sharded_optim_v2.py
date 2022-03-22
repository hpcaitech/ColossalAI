from enum import Enum
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_model._zero3_utils import cast_tensor_to_fp32
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from ._utils import has_inf_or_nan


class OptimState(Enum):
    SCALED = 1
    UNSCALED = 2


class ShardedOptimizerV2(ColossalaiOptimizer):
    """A wrapper for optimizer. `ShardedOptimizerV2` and `ShardedModelV2` implement Zero Redundancy Optimizer (ZeRO) stage 3.
    You must use `ShardedOptimizerV2` with `ShardedModelV2`.

    :param sharded_model: A sharded model initialized by class ShardedModelV2. The optimizer will use the
    shard strategy provided by sharded model to shard param fp32 tensors.
    :type sharded_model: sharded_model

    :param optimizer: A Optimizer instance.
    :type optimizer: Optimizer

    :param cpu_offload: is offloading the optimizer states to CPU.
    :type cpu_offload: bool

    :param initial_scale: initial scale used by DynamicGradScaler
    :type initial_scale: float

    :param min_scale: min scale used by DynamicGradScaler
    :type min_scale: float

    :param growth_factor: growth_factor used by DynamicGradScaler
    :type growth_factor: float

    :param backoff_factor: backoff_factor used by DynamicGradScaler
    :type backoff_factor: float

    :param growth_interval: growth_interval used by DynamicGradScaler
    :type growth_interval: float

    :param hysteresis: hysteresis used by DynamicGradScaler
    :type hysteresis: float

    :param max_scale: max_scale used by DynamicGradScaler
    :type max_scale: float

    :param dp_process_group: data paralle process group
    :type dp_process_group: Optional[ProcessGroup]

    :param mp_process_group: model paralle process group
    :type mp_process_group: Optional[ProcessGroup]
        """

    def __init__(self,
                 sharded_model: ShardedModelV2,
                 optimizer: Optimizer,
                 cpu_offload: bool = False,
                 initial_scale: float = 2**32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: float = 1000,
                 hysteresis: float = 2,
                 max_scale: int = 2**32,
                 dp_process_group: Optional[ProcessGroup] = None,
                 mp_process_group: Optional[ProcessGroup] = None) -> None:
        assert isinstance(sharded_model, ShardedModelV2), 'model must be wrapped with ShardedModel'

        super().__init__(optimizer)
        self.shard_strategy = sharded_model.shard_strategy
        self.model: ShardedModelV2 = sharded_model
        if cpu_offload and not sharded_model.cpu_offload:
            raise RuntimeError(
                f"ShardedOptimizerV2 using cpu_offload, but the sharded_model used to initialize it dose not use cpu_offload"
            )
        self.device = torch.cuda.current_device() if not cpu_offload else torch.device('cpu')
        self.optim_state: OptimState = OptimState.UNSCALED
        self.dp_process_group = dp_process_group or gpc.get_group(ParallelMode.DATA)
        self.mp_process_group = mp_process_group or gpc.get_group(ParallelMode.MODEL)
        # Grad scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale)
        self._found_overflow: Tensor = torch.FloatTensor([0]).to(torch.cuda.current_device())
        self._logger = get_dist_logger()

        # Store fp32 param shards
        self.master_params: Dict[Parameter, Tensor] = {}

        for group in self.optim.param_groups:
            for p in group['params']:
                assert hasattr(p, 'col_attr'), 'The parameter must be wrapped with ShardedParam'
                is_param_sharded = p.col_attr.sharded_data_tensor.is_sharded
                if not is_param_sharded:
                    # TODO (ver217): we may not use shard / gather here
                    # Param is no sharded, which means we use ZeRO-2 here
                    # As we only store param shard, we shard it here
                    self.shard_strategy.shard([p.col_attr.sharded_data_tensor], self.dp_process_group)
                self.master_params[p] = cast_tensor_to_fp32(p.col_attr.sharded_data_tensor.payload).to(self.device)
                if not is_param_sharded:
                    # In this branch, there's no need to shard param
                    # So we gather here
                    self.shard_strategy.gather([p.col_attr.sharded_data_tensor], self.dp_process_group)

    def step(self, *args, **kwargs):
        # unscale grads if scaled
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()

        found_inf = self._check_overflow()
        self.grad_scaler.update(found_inf)

        if found_inf:
            self._logger.info('found inf during ShardedOptimV2 step')
            self.zero_grad()
            return

        # assign master param pointers to p.data.
        # We will not trigger data copy here.
        for group in self.optim.param_groups:
            for p in group['params']:
                p.data = self.master_params[p]
                # Now p.data is sharded
                # So optimizer states are sharded naturally

        ret = self.optim.step(*args, **kwargs)

        # Copy master param data (fp32) to payload of col_attr (fp16)
        # TODO() improve efficiency by gathering tensors into a chunk and transfering
        # a chunk.
        for group in self.optim.param_groups:
            for p in group['params']:
                is_param_sharded = p.col_attr.sharded_data_tensor.is_sharded
                if not is_param_sharded:
                    # We use ZeRO-2 here
                    # The `p.col_attr.sharded_data_tensor` saves full fp16 param
                    # But we only have updated fp32 param shard here
                    # So we first shard full fp16 param and copy fp32 param shard to it
                    # Then we will gather them
                    self.shard_strategy.shard([p.col_attr.sharded_data_tensor], self.dp_process_group)
                # We have to use `copy_payload` instead of `reset_payload`
                # Since p.data is fp32 and p.col_attr.sharded_data_tensor is fp16

                # TODO() optimize this line CPU (fp32) -> GPU (fp16)
                p.col_attr.sharded_data_tensor.copy_payload(p.data)

                if not is_param_sharded:
                    # We gather full fp16 param here
                    self.shard_strategy.gather([p.col_attr.sharded_data_tensor], self.dp_process_group)
                p.data = p.col_attr.sharded_data_tensor.payload
        return ret

    def backward(self, loss: Tensor) -> None:
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        self.model.backward(loss)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor) -> None:
        self.model.backward_by_grad(tensor, grad)

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        return super().clip_grad_norm(model, max_norm)

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(0.0)

        # check for overflow
        for group in self.optim.param_groups:
            for p in group['params']:
                if has_inf_or_nan(p.grad):
                    self._found_overflow.fill_(1.0)
                    break

        # all-reduce across dp group
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        # all-reduce over model parallel group
        dist.all_reduce(self._found_overflow, op=dist.ReduceOp.MAX, group=self.mp_process_group)

        return self._found_overflow.item() > 0

    def _unscale_grads(self):
        assert self.optim_state == OptimState.SCALED
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)
        self.optim_state = OptimState.UNSCALED

    def zero_grad(self, *args, **kwargs):
        # We must set grad to None
        # Because we will judge whether local grad accumulation
        # is enabled by wheter grad is None
        self.optim.zero_grad(set_to_none=True)

    def sync_grad(self):
        pass

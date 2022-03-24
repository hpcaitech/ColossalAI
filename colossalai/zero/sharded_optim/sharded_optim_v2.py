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
from colossalai.zero.sharded_optim._utils import has_inf_or_nan
from colossalai.utils.memory_utils.utils import colo_model_data_tensor_move


class OptimState(Enum):
    SCALED = 1
    UNSCALED = 2


class ShardedOptimizerV2(ColossalaiOptimizer):
    """A wrapper for optimizer. `ShardedOptimizerV2` and `ShardedModelV2` implement Zero Redundancy Optimizer (ZeRO).
    By default the ZeRO optimizer stage 3 offload Optimizer States on CPU.
    We apply the Device-aware Operator Placement technique for OS placement from the following paper.
    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818
    GPU margin space is the remaining space after removing peak non-model data from the overall GPU memory,
    which is detected by a runtime memory tracer. 
    We place as many OS chunks in the margin space as possible. 
    The size of margin space can be controlled by `gpu_margin_mem_ratio`
    If it is set as 0.0, it is the same as classical ZeRO optimizer.

    NOTE() You must use `ShardedOptimizerV2` with `ShardedModelV2`.

    Args:
        sharded_model (ShardedModelV2): A sharded model initialized by class ShardedModelV2. The optimizer will use the
            shard strategy provided by sharded model to shard param fp32 tensors.
        optimizer (Optimizer): An Optimizer instance.
        cpu_offload (bool, optional): Is offloading the optimizer states to CPU.. Defaults to False.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward) 
            which will be used when using hybrid CPU optimizer. Defaults to 0.0.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**32.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        dp_process_group (Optional[ProcessGroup], optional): data paralle process group. Defaults to None.
        mp_process_group (Optional[ProcessGroup], optional): model paralle process group. Defaults to None.
    """

    def __init__(self,
                 sharded_model: ShardedModelV2,
                 optimizer: Optimizer,
                 cpu_offload: bool = False,
                 gpu_margin_mem_ratio: float = 0.0,
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
        self.gpu_margin_mem_ratio: float = float(gpu_margin_mem_ratio)
        assert 0.0 <= self.gpu_margin_mem_ratio <= 1.0, f'gpu_margin_mem_ratio must >=0.0 and <=1.0'
        # Only move fp32 shards from CPU to GPU when user allows and inner optimizer is valid
        # Inner optimizer must support optimizing hybrid (CPU and CUDA) tensors,
        # and it must set `num_fp32_shards_per_param` correctly
        self._should_move_fp32_shards_h2d: bool = cpu_offload and self.gpu_margin_mem_ratio > 0.0 and getattr(
            optimizer, 'num_fp32_shards_per_param', 0) >= 2
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
        self._maybe_move_fp32_shards()

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
                colo_model_data_tensor_move(p, p.col_attr.sharded_data_tensor)

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

    def _maybe_move_fp32_shards(self):
        if self._should_move_fp32_shards_h2d:
            self._should_move_fp32_shards_h2d = False
            available_cuda_margin_mem = self.model.cuda_margin_space * self.gpu_margin_mem_ratio
            fp32_shards_available_cuda_margin_mem = available_cuda_margin_mem / self.optim.num_fp32_shards_per_param
            fp32_shards_used_cuda_margin_mem = 0
            for group in self.optim.param_groups:
                for p in group['params']:
                    shard_mem = self.master_params[p].numel() * self.master_params[p].element_size()
                    if fp32_shards_used_cuda_margin_mem + shard_mem < fp32_shards_available_cuda_margin_mem:
                        self.master_params[p] = self.master_params[p].to(torch.cuda.current_device())
                        p.grad.data = p.grad.data.to(torch.cuda.current_device())
                        p.col_attr.offload_grad = False
                        fp32_shards_used_cuda_margin_mem += shard_mem

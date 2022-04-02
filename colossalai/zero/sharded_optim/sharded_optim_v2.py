from enum import Enum
from os import stat
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.utils.memory_tracer.model_data_memtracer import \
    GLOBAL_MODEL_DATA_TRACER
from colossalai.zero.shard_utils.tensor_utils import (colo_model_data_tensor_move_inline, colo_model_tensor_clone,
                                                      colo_tensor_mem_usage)
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_model._utils import cast_tensor_to_fp32
from colossalai.zero.sharded_optim._utils import has_inf_or_nan
from colossalai.zero.sharded_param.tensorful_state import (StatefulTensor, TensorState)
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from torch.optim import Optimizer


class OptimState(Enum):
    SCALED = 1
    UNSCALED = 2


class ShardedOptimizerV2(ColossalaiOptimizer):
    """A wrapper for optimizer. ``ShardedOptimizerV2`` and ``ShardedModelV2`` implement Zero Redundancy Optimizer (ZeRO).

    By default the ZeRO optimizer stage 3 offload Optimizer States on CPU.

    We apply the Device-aware Operator Placement technique for OS placement from the following paper.

    `PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management`_

    GPU margin space is the remaining space after removing peak non-model data from the overall GPU memory,
    which is detected by a runtime memory tracer. 

    We place as many OS chunks in the margin space as possible. 

    The size of margin space can be controlled by ``gpu_margin_mem_ratio``.
    If it is set as ``0.0``, it is the same as classical ZeRO optimizer.

    Note:
        You must use ``ShardedOptimizerV2`` with ``ShardedModelV2``.

    Note:
        Make sure you enable ``use_memory_tracer`` in ``ShardedModelV2``,
        if you set ``gpu_margin_mem_ratio > 0``.

    Args:
        sharded_model (ShardedModelV2): A sharded model initialized by class ShardedModelV2. The optimizer will use the
            shard strategy provided by sharded model to shard param fp32 tensors.
        optimizer (Optimizer): An Optimizer instance.
        cpu_offload (bool, optional): Is offloading the optimizer states to CPU.. Defaults to False.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward) 
            which will be used when using hybrid CPU optimizer. 
            Make sure `reuse_fp16_shard` is enabled in `ShardedModelV2`, if `gpu_margin_mem_ratio` > `0.0`.
            Defaults to 0.0.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**32.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        keep_unsharded (bool, optional): if True, optimizer won't shard unsharded parameters.
            In Zero-2, set keep_unsharded to False.
            In Zero-3, set keep_unsharded to True.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        dp_process_group (Optional[ProcessGroup], optional): data paralle process group. Defaults to None.
        mp_process_group (Optional[ProcessGroup], optional): model paralle process group. Defaults to None.

    .. _PatrickStar\: Parallel Training of Pre-trained Models via Chunk-based Memory Management:
        https://arxiv.org/abs/2108.05818
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
                 keep_unsharded: bool = False,
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
        self._logger = get_dist_logger("ShardedOptimizerV2")

        assert not (keep_unsharded and self._should_move_fp32_shards_h2d), \
            "Keeping unsharded parameters can't be used with hybrid OS placement right now."
        self.keep_unshard = keep_unsharded

        # Store fp32 param shards
        self._register_master_weight()

        self._logger.debug(f"After init ShardedOptimizerV2 consumes {self.get_memory_usage()[0]/1e6} MB CUDA Memory!",
                           ranks=[0])

        self._use_memory_tracer = self.model.use_memory_tracer
        if self._use_memory_tracer:
            GLOBAL_MODEL_DATA_TRACER.register_optimizer(self)

        self._use_memory_tracer = self.model.use_memory_tracer
        if self._use_memory_tracer:
            GLOBAL_MODEL_DATA_TRACER.register_optimizer(self)

    def get_memory_usage(self) -> Tuple[int, int]:
        """ Get the memory usage of the optimizer. Including master_params (param fp32),
        momentum (``self.state[p]['exp_avg']``) variance (``self.state[p]['exp_avg_sq']``)

        Returns:
            Tuple[int, int]: cuda/cpu memory usage in Byte.
        """
        cuda_use = 0
        cpu_use = 0

        def update_mem_use(t):
            nonlocal cuda_use
            nonlocal cpu_use
            t_cuda_use, t_cpu_use = colo_tensor_mem_usage(t)
            cuda_use += t_cuda_use
            cpu_use += t_cpu_use

        for _, p_fp32 in self.master_params.items():
            update_mem_use(p_fp32)
        for group in self.optim.param_groups:
            for p in group['params']:
                state = self.optim.state[p]
                for k, v in state.items():
                    update_mem_use(v)

        return cuda_use, cpu_use

    def step(self, *args, **kwargs):
        self._prepare_grads()
        self._maybe_move_fp32_shards()

        # unscale grads if scaled
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()

        found_inf = self._check_overflow()
        self.grad_scaler.update(found_inf)

        if found_inf:
            self._logger.warning('found inf during ShardedOptimV2 step')
            self._zero_grad(recover_data=True)
            return

        self._prepare_data()

        self._logger.debug(
            f"Before step ShardedOptimizerV2 consumes {self.get_memory_usage()[0]/1e6} MB CUDA Memory, {self.get_memory_usage()[1]/1e6} MB CUDA Memory!",
            ranks=[0])

        ret = self.optim.step(*args, **kwargs)

        self._logger.debug(
            f"After step ShardedOptimizerV2 consumes {self.get_memory_usage()[0]/1e6} MB CUDA Memory, {self.get_memory_usage()[1]/1e6} MB CUDA Memory!",
            ranks=[0])
        self._write_back_data()
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
        self._zero_grad()

    def _zero_grad(self, recover_data: bool = False):
        """zero grad and maybe recover fp16 params
        When `reuse_fp16_shard` is enabled,
        p.colo_attr.sharded_data_tensor stores grad here.
        We have to recover them from fp32 params.

        Args:
            recover_data (bool, optional): Whether to recover fp16 param from fp32 param. Defaults to False.
        """
        # We must set grad to None
        # Because grad here is sharded
        # But next backward pass will create a full grad first
        # Which leads to wrong accumulation
        self.optim.zero_grad(set_to_none=True)
        for group in self.optim.param_groups:
            for p in group['params']:
                # p.colo_attr.sharded_data_tensor stores grad now
                # we have to recover fp16 param
                reuse_fp16_shard = p.colo_attr.saved_grad.data_ptr() == p.colo_attr.sharded_data_tensor.data_ptr()
                p.colo_attr.saved_grad.set_null()
                if recover_data and reuse_fp16_shard:
                    p.colo_attr.sharded_data_tensor.reset_payload(
                        colo_model_tensor_clone(self.master_params[p].payload.half(), torch.cuda.current_device()))

    def sync_grad(self):
        pass

    def _register_master_weight(self):
        self.master_params: Dict[Parameter, StatefulTensor] = {}
        for group in self.optim.param_groups:
            for p in group['params']:
                assert hasattr(p, 'colo_attr'), 'The parameter must be wrapped with ShardedParam'
                is_param_sharded = p.colo_attr.sharded_data_tensor.is_sharded
                if not is_param_sharded and not self.keep_unshard:
                    # Please use keep_unsharded to control whether shard unsharded paramters
                    # As we only store param shard, we shard it here
                    self.shard_strategy.shard([p.colo_attr.sharded_data_tensor], self.dp_process_group)
                self.master_params[p] = StatefulTensor(
                    cast_tensor_to_fp32(p.colo_attr.sharded_data_tensor.payload).to(self.device))
                if not is_param_sharded and not self.keep_unshard:
                    # In this branch, there's no need to shard param
                    # So we gather here
                    self.shard_strategy.gather([p.colo_attr.sharded_data_tensor], self.dp_process_group)

    def _maybe_move_fp32_shards(self):
        if self._should_move_fp32_shards_h2d:
            self._should_move_fp32_shards_h2d = False
            available_cuda_margin_mem = self.model.cuda_margin_space * self.gpu_margin_mem_ratio
            fp32_shards_available_cuda_margin_mem = available_cuda_margin_mem / self.optim.num_fp32_shards_per_param
            fp32_shards_used_cuda_margin_mem = 0
            for group in self.optim.param_groups:
                for p in group['params']:
                    shard_mem = self.master_params[p].payload.numel() * self.master_params[p].payload.element_size()
                    if fp32_shards_used_cuda_margin_mem + shard_mem < fp32_shards_available_cuda_margin_mem:
                        colo_model_data_tensor_move_inline(self.master_params[p], torch.cuda.current_device())
                        p.grad.data = p.grad.data.to(torch.cuda.current_device())
                        p.colo_attr.offload_grad = False
                        fp32_shards_used_cuda_margin_mem += shard_mem

    def _prepare_grads(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                p.colo_attr.saved_grad.trans_state(TensorState.COMPUTE)
                # FIXME(ver217): p.data here is an empty tensor on CUDA and has no useful infomation
                # If we change p.grad directly
                # it may raise error because of different shape/dtype/device of p.data and p.grad
                # We just set p.data = p.colo_attr.saved_grad.payload here
                p.data = p.colo_attr.saved_grad.payload
                p.grad = p.colo_attr.saved_grad.payload
                # Set p.data to empty tensor, in case of memory leaking
                p.colo_attr.remove_torch_payload()

    def _prepare_data(self):
        # assign master param pointers to p.data.
        # We will not trigger data copy here.
        for group in self.optim.param_groups:
            for p in group['params']:
                self.master_params[p].trans_state(TensorState.COMPUTE)
                p.data = self.master_params[p].payload
                # Now p.data is sharded
                # So optimizer states are sharded naturally

    def _write_back_data(self):
        # Copy master param data (fp32) to payload of colo_attr (fp16)
        # TODO() improve efficiency by gathering tensors into a chunk and transfering
        # a chunk.
        for group in self.optim.param_groups:
            for p in group['params']:
                is_param_sharded = p.colo_attr.sharded_data_tensor.is_sharded
                if not is_param_sharded and not self.keep_unshard:
                    # We use ZeRO-2 here
                    # The `p.colo_attr.sharded_data_tensor` saves full fp16 param
                    # But we only have updated fp32 param shard here
                    # So we first shard full fp16 param and copy fp32 param shard to it
                    # Then we will gather them
                    self.shard_strategy.shard([p.colo_attr.sharded_data_tensor], self.dp_process_group)
                # We have to use `copy_payload` instead of `reset_payload`
                # Since p.data is fp32 and p.colo_attr.sharded_data_tensor is fp16

                # TODO() optimize this line CPU (fp32) -> GPU (fp16)
                p.colo_attr.sharded_data_tensor.reset_payload(
                    colo_model_tensor_clone(p.half(), torch.cuda.current_device()))

                if not is_param_sharded and not self.keep_unshard:
                    # We gather full fp16 param here
                    self.shard_strategy.gather([p.colo_attr.sharded_data_tensor], self.dp_process_group)
                p.data = p.colo_attr.sharded_data_tensor.payload
                self.master_params[p].trans_state(TensorState.HOLD)
                p.colo_attr.saved_grad.set_null()

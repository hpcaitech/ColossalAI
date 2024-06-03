# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.interface import OptimizerWrapper
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.zero.gemini.stateful_tensor import StatefulTensor, TensorState
from colossalai.legacy.zero.gemini.tensor_placement_policy import AutoTensorPlacementPolicy
from colossalai.legacy.zero.gemini.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.legacy.zero.sharded_model import ShardedModelV2
from colossalai.legacy.zero.sharded_model._utils import cast_tensor_to_fp32
from colossalai.logging import get_dist_logger


class OptimState(Enum):
    SCALED = 1
    UNSCALED = 2


class ShardedOptimizerV2(OptimizerWrapper):
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
        Make sure you set ``tensor_placement_policy`` in ``ShardedModelV2`` to `"auto"`,
        if you set ``gpu_margin_mem_ratio > 0``.

    Args:
        sharded_model (ShardedModelV2): A sharded model initialized by class ShardedModelV2. The optimizer will use the
            shard strategy provided by sharded model to shard param fp32 tensors.
        optimizer (Optimizer): An Optimizer instance.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used when using hybrid CPU optimizer.
            This argument is meaningless when `tensor_placement_policy` of `ShardedModelV2` is not "auto".
            Defaults to 0.0.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**32.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        dp_process_group (Optional[ProcessGroup], optional): data parallel process group. Defaults to None.
        mp_process_group (Optional[ProcessGroup], optional): model parallel process group. Defaults to None.

    .. _PatrickStar\: Parallel Training of Pre-trained Models via Chunk-based Memory Management:
        https://arxiv.org/abs/2108.05818
    """

    def __init__(
        self,
        sharded_model: ShardedModelV2,
        optimizer: Optimizer,
        gpu_margin_mem_ratio: float = 0.0,
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        dp_process_group: Optional[ProcessGroup] = None,
        mp_process_group: Optional[ProcessGroup] = None,
        verbose: bool = False,
    ) -> None:
        assert isinstance(sharded_model, ShardedModelV2), "model must be wrapped with ShardedModel"
        assert not isinstance(optimizer, ShardedOptimizerV2), "Nested ShardedOptimizerV2 is not supported."

        super().__init__(optimizer)
        self.shard_strategy = sharded_model.shard_strategy
        self.model: ShardedModelV2 = sharded_model
        self.bf16 = sharded_model.bf16

        self.gpu_margin_mem_ratio: float = float(gpu_margin_mem_ratio)
        assert 0.0 <= self.gpu_margin_mem_ratio <= 1.0, f"gpu_margin_mem_ratio must >=0.0 and <=1.0"
        # Only move fp32 shards from CPU to GPU when user allows and inner optimizer is valid
        # Inner optimizer must support optimizing hybrid (CPU and CUDA) tensors,
        # and it must set `num_fp32_shards_per_param` correctly
        self._should_move_fp32_shards_h2d: bool = (
            sharded_model.cpu_offload
            and self.gpu_margin_mem_ratio > 0.0
            and getattr(optimizer, "num_fp32_shards_per_param", 0) >= 2
        )
        self.device = sharded_model._tensor_placement_policy.device or torch.device("cpu")
        self.optim_state: OptimState = OptimState.UNSCALED
        self.dp_process_group = dp_process_group or gpc.get_group(ParallelMode.DATA)
        self.mp_process_group = mp_process_group or gpc.get_group(ParallelMode.MODEL)
        # Grad scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
        )
        self._found_overflow: Tensor = torch.IntTensor([0]).to(torch.cuda.current_device())
        self._logger = get_dist_logger("ShardedOptimizerV2")
        self._verbose = verbose
        self._grad_prepared: bool = (
            False  # this should be set to true when _prepare_grads() and reset to false when backward
        )

        # Store fp32 param shards
        self._register_master_weight()
        if self.gpu_margin_mem_ratio != 0.0 and not isinstance(
            sharded_model._tensor_placement_policy, AutoTensorPlacementPolicy
        ):
            self._logger.warning(
                f'gpu_margin_mem_ratio is meaningless when tensor_placement_policy is not "auto"', ranks=[0]
            )

        if self._verbose:
            self._logger.debug(
                f"After init ShardedOptimizerV2 consumes {self.get_memory_usage()[0] / 1e6} MB CUDA Memory!", ranks=[0]
            )

        self._use_memory_tracer = self.model.use_memory_tracer

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def get_memory_usage(self) -> Tuple[int, int]:
        """Get the memory usage of the optimizer. Including master_params (param fp32),
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
            for p in group["params"]:
                state = self.optim.state[p]
                for k, v in state.items():
                    update_mem_use(v)

        return cuda_use, cpu_use

    def zero_grad(self, *args, **kwargs):
        self._zero_grad()

    def backward(self, loss: Tensor) -> None:
        if not self.bf16:
            loss = self.loss_scale * loss
            self.optim_state = OptimState.SCALED
        self._grad_prepared = False
        self.model.backward(loss)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor) -> None:
        # This function is called except the last stage of pipeline parallel
        # It receives the scaled grad from the previous rank
        # No need to scale the grad again
        # Need to unscale when optimizing
        if not self.bf16:
            self.optim_state = OptimState.SCALED
        self._grad_prepared = False
        self.model.backward_by_grad(tensor, grad)

    def clip_grad_norm(self, model: nn.Module, max_norm: float):
        self._prepare_grads()
        if not self.bf16 and self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        return super().clip_grad_norm(model, max_norm)

    def step(self, *args, **kwargs):
        self._prepare_grads()
        # unscale grads if scaled
        if not self.bf16 and self.optim_state == OptimState.SCALED:
            self._unscale_grads()

        self._maybe_move_fp32_shards()
        if not self.bf16:
            found_inf = self._check_overflow()
            self.grad_scaler.update(found_inf)

            if found_inf:
                self._logger.warning("found inf during ShardedOptimV2 step")
                self._zero_grad(recover_data=True)
                return

        self._point_param_fp16_to_master_param()

        if self._verbose:
            gpu_mem, cpu_mem = self.get_memory_usage()
            self._logger.debug(
                f"Before step ShardedOptimizerV2 consumes {gpu_mem / 1e6} MB CUDA Memory, {cpu_mem / 1e6} MB CUDA Memory!",
                ranks=[0],
            )
        ret = self.optim.step(*args, **kwargs)

        if self._verbose:
            gpu_mem, cpu_mem = self.get_memory_usage()
            self._logger.debug(
                f"After step ShardedOptimizerV2 consumes {gpu_mem / 1e6} MB CUDA Memory, {cpu_mem / 1e6} MB CUDA Memory!",
                ranks=[0],
            )

        self._copy_master_model_to_model_fp16()
        return ret

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.model.overflow_counter)

        # all-reduce across dp group
        dist.all_reduce(self._found_overflow, group=self.dp_process_group)

        # all-reduce over model parallel group
        dist.all_reduce(self._found_overflow, group=self.mp_process_group)

        return self._found_overflow.item() > 0

    def _unscale_grads(self):
        assert self.optim_state == OptimState.SCALED
        for group in self.optim.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)
        self.optim_state = OptimState.UNSCALED

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
            for p in group["params"]:
                # p.colo_attr.sharded_data_tensor stores grad now
                # we have to recover fp16 param
                reuse_fp16_shard = p.colo_attr.sharded_data_tensor.payload_size == 0
                if recover_data and reuse_fp16_shard:
                    self._copy_master_param_to_param_fp16(p)
                else:
                    # release saved gradient
                    p.colo_attr.saved_grad.set_null()
        self.model.overflow_counter = 0  # set overflow counter to zero

    def sync_grad(self):
        pass

    def _register_master_weight(self):
        self.master_params: Dict[Parameter, StatefulTensor] = {}
        for group in self.optim.param_groups:
            for p in group["params"]:
                assert hasattr(p, "colo_attr"), "The parameter must be wrapped with ShardedParam"
                shard_flag = not p.colo_attr.sharded_data_tensor.is_sharded and p.colo_attr.is_replicated
                if shard_flag:
                    # we always shard replicated parameters
                    self.shard_strategy.shard([p.colo_attr.sharded_data_tensor], self.dp_process_group)
                self.master_params[p] = StatefulTensor(cast_tensor_to_fp32(p.colo_attr.data_payload.to(self.device)))
                if shard_flag:
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
                for p in group["params"]:
                    if p.colo_attr.saved_grad.is_null():
                        continue
                    shard_mem = self.master_params[p].payload.numel() * self.master_params[p].payload.element_size()
                    if fp32_shards_used_cuda_margin_mem + shard_mem < fp32_shards_available_cuda_margin_mem:
                        colo_model_data_tensor_move_inline(self.master_params[p], torch.cuda.current_device())
                        colo_model_data_tensor_move_inline(p.colo_attr.saved_grad, torch.cuda.current_device())
                        p.colo_attr.offload_grad = False
                        fp32_shards_used_cuda_margin_mem += shard_mem
                        state = self.optim.state[p]
                        for k, v in state.items():
                            if isinstance(v, Tensor):
                                state[k] = v.cuda()

    def _prepare_grads(self):
        if self._grad_prepared:
            return
        for group in self.optim.param_groups:
            for p in group["params"]:
                if p.colo_attr.saved_grad.is_null():
                    continue
                p.colo_attr.saved_grad.trans_state(TensorState.COMPUTE)
                # If reuse_fp16_shard, grad fp16 which wasn't be offloaded may be evicted to CPU
                if not p.colo_attr.offload_grad:
                    colo_model_data_tensor_move_inline(p.colo_attr.saved_grad, torch.cuda.current_device())
                # FIXME(ver217): p.data here is an empty tensor on CUDA and has no useful information
                # If we change p.grad directly
                # it may raise error because of different shape/dtype/device of p.data and p.grad
                # We just set p.data = p.colo_attr.saved_grad.payload here
                p.data = p.colo_attr.grad_payload
                p.grad = p.colo_attr.grad_payload
                # Set p.data to empty tensor, in case of memory leaking
                p.colo_attr.set_data_none()
        self._grad_prepared = True

    def _point_param_fp16_to_master_param(self):
        # assign master param pointers to p.data.
        # We will not trigger data copy here.
        for group in self.optim.param_groups:
            for p in group["params"]:
                self.master_params[p].trans_state(TensorState.COMPUTE)
                p.data = self.master_params[p].payload
                # Now p.data is sharded
                # So optimizer states are sharded naturally

    def _copy_master_model_to_model_fp16(self):
        # Copy master param data (fp32) to payload of colo_attr (fp16)
        # TODO() improve efficiency by gathering tensors into a chunk and transferring
        # a chunk.
        for group in self.optim.param_groups:
            for p in group["params"]:
                self._copy_master_param_to_param_fp16(p)

    def _copy_master_param_to_param_fp16(self, p):
        # flush gradient
        if p.colo_attr.sharded_data_tensor.payload_size == 0:
            # here reuse_fp16_shard is True
            # in order to use copy below, we should give sharded data tensor a payload
            p.colo_attr.sharded_data_tensor.payload_relay(p.colo_attr.saved_grad)
        else:
            p.colo_attr.saved_grad.set_null()

        p.data = self.master_params[p].payload

        # we need to allocate new memory for keep_not_shard parameters
        # in order to use copy, otherwise, the sizes of tensor is not compatible
        if p.colo_attr.data_payload.numel() != p.data.numel():
            p.colo_attr.data_payload_reset(
                torch.empty(p.data.shape, dtype=p.colo_attr.data_payload.dtype, device=p.colo_attr.data_payload.device)
            )

        # TODO() optimize this line CPU (fp32) -> GPU (fp16)
        half_dtype = torch.bfloat16 if self.bf16 else torch.float16
        p.colo_attr.sharded_data_tensor.payload_copy(p.to(half_dtype).detach())
        p.colo_attr.set_data_none()

        if p.colo_attr.keep_not_shard and p.colo_attr.is_replicated:
            # We gather full fp16 param here
            p.colo_attr.sharded_data_tensor.is_sharded = True  # since only gradient is sharded, we should set to True
            self.shard_strategy.gather([p.colo_attr.sharded_data_tensor], self.dp_process_group)

        self.master_params[p].trans_state(TensorState.HOLD)

    def state_dict(self):
        optim_state_dict = super().state_dict()
        scaler_state_dict = self.grad_scaler.state_dict()
        optim_state_dict["scaler"] = scaler_state_dict
        return optim_state_dict

    def load_state_dict(self, *args, **kwargs):
        if "scaler" not in args[0]:
            self._logger.warning("Missing scaler when loading optimizer state dict", ranks=[0])
        else:
            scaler_state_dict = args[0].pop("scaler")
            self.grad_scaler.load_state_dict(scaler_state_dict)
        super().load_state_dict(*args, **kwargs)
        for group in self.optim.param_groups:
            for p in group["params"]:
                state = self.optim.state[p]
                for k, v in state.items():
                    if isinstance(v, Tensor):
                        state[k] = v.to(dtype=self.master_params[p].dtype, device=self.master_params[p].device)

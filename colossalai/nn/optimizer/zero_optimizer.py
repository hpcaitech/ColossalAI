import math
import warnings
from enum import Enum
from typing import Any, Dict, Set, Tuple

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import Optimizer

from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.gemini.chunk import Chunk, ChunkManager
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer, CPUAdam, FusedAdam, HybridAdam
from colossalai.nn.parallel.data_parallel import ZeroDDP
from colossalai.utils import disposable, get_current_device, is_ddp_ignored

_AVAIL_OPTIM_LIST = {FusedAdam, CPUAdam, HybridAdam}


class OptimState(Enum):
    SCALED = 0
    UNSCALED = 1


class ZeroOptimizer(ColossalaiOptimizer):
    """A wrapper for optimizer. ``ZeroDDP`` and ``ZeroOptimizer`` implement Zero Redundancy Optimizer (ZeRO state-3).

    Note:
        You must use ``ZeroDDP`` with ``ZeroOptimizer``.

    Note:
        Make sure you set ``placement_policy`` of ``GeminiManager`` to `"auto"`,
        if you set ``gpu_margin_mem_ratio > 0``.

    Args:
        optim (Optimizer): An Optimizer instance.
        module (ZeroDDP): A ``ZeroDDP`` instance.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used when using hybrid CPU optimizer.
            This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".
            Defaults to 0.0.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**32.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        """

    def __init__(self,
                 optim: Optimizer,
                 module: ZeroDDP,
                 gpu_margin_mem_ratio: float = 0.0,
                 initial_scale: float = 2**32,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32,
                 clipping_norm: float = 0.0,
                 norm_type: float = 2.0,
                 **defaults: Any):
        super().__init__(optim)
        assert isinstance(module, ZeroDDP)
        assert type(optim) in _AVAIL_OPTIM_LIST, "you should use the optimizer in the available list"
        self.module = module
        self.gemini_manager = module.gemini_manager
        self.chunk_manager: ChunkManager = self.gemini_manager.chunk_manager
        self.optim_state = OptimState.UNSCALED
        self.param_to_range: Dict[Parameter, Tuple[int, int]] = dict()
        self.param_to_chunk32: Dict[Parameter, Chunk] = dict()
        self.chunk16_set: Set[Chunk] = set()
        self.clipping_flag = clipping_norm > 0.0
        self.max_norm = clipping_norm

        if self.clipping_flag:
            assert norm_type == 2.0, "ZeroOptimizer only supports L2 norm now"

        ddp_param_list = []
        for name, param in module.named_parameters():
            if is_ddp_ignored(param):
                if param.requires_grad:
                    warnings.warn(f"Parameter `{name}` is ignored by DDP but requires gradient! "
                                  "You should handle its optimizer update by yourself!")
            else:
                ddp_param_list.append(param)

        for p, fp32_p in zip(ddp_param_list, module.fp32_params):
            chunk_16 = self.chunk_manager.get_chunk(p)
            if chunk_16 not in self.chunk16_set:
                chunk_16.l2_norm_flag = self.clipping_flag
                self.chunk16_set.add(chunk_16)

        self.__init__optimizer()

        # Grad scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale)
        self._found_overflow: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=get_current_device())
        self._logger = get_dist_logger()

        self.gpu_margin_mem_ratio: float = float(gpu_margin_mem_ratio)
        assert 0.0 <= self.gpu_margin_mem_ratio <= 1.0, f'gpu_margin_mem_ratio must >=0.0 and <=1.0'
        # Only move fp32 shards from CPU to GPU when user allows and inner optimizer is valid
        # Inner optimizer must support optimizing hybrid (CPU and CUDA) tensors,
        # and it must set `num_fp32_shards_per_param` correctly
        self._should_move_fp32_params_h2d: bool = self.gemini_manager.is_cuda_margin_mem_avail and self.gpu_margin_mem_ratio > 0.0 and getattr(
            optim, 'num_fp32_shards_per_param', 0) >= 2
        if self.gpu_margin_mem_ratio > 0.0 and not self.gemini_manager.is_cuda_margin_mem_avail:
            self._logger.warning(f'gpu_margin_mem_ratio is meaningless when placement_policy is not "auto"', ranks=[0])

        self._register_states = disposable(self._register_states_)

    def _set_grad_ptr(self):
        for group in self.param_groups:
            for fake_param in group['params']:
                chunk32 = self.param_to_chunk32[fake_param]
                begin, end = self.param_to_range[fake_param]
                chunk16 = chunk32.paired_chunk

                fake_param.data = chunk16.payload[begin:end]
                fake_param.grad = fake_param.data
                fake_param.data = chunk32.payload[begin:end]

    def _update_fp16_params(self):
        none_tensor = torch.empty([0])
        for group in self.param_groups:
            for fake_param in group['params']:
                assert fake_param.grad is None
                fake_param.data = none_tensor

        for chunk16 in self.chunk16_set:
            chunk16.optim_update()

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter)

        # all-reduce across global group
        dist.all_reduce(self._found_overflow)

        return self._found_overflow.item() > 0

    def _clear_global_norm(self) -> None:
        for c16 in self.chunk16_set:
            c16.l2_norm = None

    def _calc_global_norm(self) -> float:
        norm_sqr: float = 0.0
        group_to_norm = dict()
        for c16 in self.chunk16_set:
            assert c16.l2_norm is not None

            if c16.is_gathered:
                norm_sqr += c16.l2_norm
            else:
                # this chunk is sharded, use communication to collect total norm
                if c16.torch_pg not in group_to_norm:
                    group_to_norm[c16.torch_pg] = 0.0
                group_to_norm[c16.torch_pg] += c16.l2_norm

            c16.l2_norm = None    # clear l2 norm

        comm_buffer = torch.zeros(1, dtype=torch.float, device=get_current_device())
        for group, part_norm in group_to_norm.items():
            comm_buffer.fill_(part_norm)
            dist.all_reduce(comm_buffer, group=group)
            norm_sqr += comm_buffer.item()

        global_norm = math.sqrt(norm_sqr)
        return global_norm

    def _get_combined_scale(self):
        loss_scale = 1

        if self.optim_state == OptimState.SCALED:
            loss_scale = self.loss_scale
            self.optim_state = OptimState.UNSCALED

        combined_scale = loss_scale
        if self.clipping_flag:
            total_norm = self._calc_global_norm()
            clip = ((total_norm / loss_scale) + 1e-6) / self.max_norm
            if clip > 1:
                combined_scale = clip * loss_scale

        if combined_scale == 1:
            return -1
        else:
            return combined_scale

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = 0
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        self._maybe_move_fp32_params()
        self._set_grad_ptr()

        found_inf = self._check_overflow()
        if found_inf:
            self.optim_state = OptimState.UNSCALED    # no need to unscale grad
            self.grad_scaler.update(found_inf)    # update gradient scaler
            self._logger.info(f'Found overflow. Skip step')
            self._clear_global_norm()    # clear recorded norm
            self.zero_grad()    # reset all gradients
            self._update_fp16_params()
            return

        # get combined scale. combined scale = loss scale * clipping norm
        # so that gradient = gradient / combined scale
        combined_scale = self._get_combined_scale()
        self.grad_scaler.update(found_inf)

        ret = self.optim.step(div_scale=combined_scale, *args, **kwargs)
        self._register_states()
        self.zero_grad()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float, norm_type: float = 2.0):
        raise NotImplementedError

    def backward(self, loss: torch.Tensor):
        loss = self.loss_scale * loss
        self.optim_state = OptimState.SCALED
        self.module.backward(loss)

    def backward_by_grad(self, tensor: torch.Tensor, grad: torch.Tensor):
        # This function is called except the last stage of pipeline parallel
        # It receives the scaled grad from the previous rank
        # No need to scale the grad again
        # Need to unscale when optimizing
        self.optim_state = OptimState.SCALED
        self.module.backward_by_grad(tensor, grad)

    def _maybe_move_fp32_params(self):
        if self._should_move_fp32_params_h2d:
            self._should_move_fp32_params_h2d = False
            available_cuda_margin_mem = self.gemini_manager.cuda_margin_mem * self.gpu_margin_mem_ratio
            fp32_params_available_cuda_margin_mem = available_cuda_margin_mem / self.optim.num_fp32_shards_per_param
            fp32_params_used_cuda_margin_mem = 0

            for group in self.param_groups:
                for fake_param in group['params']:
                    chunk32 = self.param_to_chunk32[fake_param]
                    chunk16 = chunk32.paired_chunk

                    if chunk32.device_type == 'cuda':
                        continue

                    if fp32_params_used_cuda_margin_mem + chunk32.payload_mem < fp32_params_available_cuda_margin_mem:
                        self.chunk_manager.move_chunk(chunk32, get_current_device())
                        # stores grad now
                        self.chunk_manager.move_chunk(chunk16, get_current_device())
                        self.module.set_chunk_grad_device(chunk16, get_current_device())
                        fp32_params_used_cuda_margin_mem += chunk32.payload_mem

            for group in self.param_groups:
                for fake_param in group['params']:
                    chunk32 = self.param_to_chunk32[fake_param]
                    if chunk32.device_type == 'cuda':
                        state = self.optim.state[fake_param]
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(get_current_device())

    def _register_states_(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                state = self.optim.state[p]
                for val in state.values():
                    if isinstance(val, torch.Tensor):
                        self.chunk_manager.add_extern_static_tensor(val)

    def __init__optimizer(self):

        def get_range_pair(local_chunk: Chunk, local_param: Parameter):
            param_info = local_chunk.tensors_info[local_param]
            if local_chunk.keep_gathered:
                return param_info.offset, param_info.end
            begin = max(0, param_info.offset - local_chunk.shard_begin)
            end = min(local_chunk.shard_size, param_info.end - local_chunk.shard_begin)
            return begin, end

        for group in self.optim.param_groups:
            fake_params_list = list()

            for param in group['params']:
                if is_ddp_ignored(param):
                    continue
                chunk16 = self.chunk_manager.get_chunk(param)
                range_pair = get_range_pair(chunk16, param)
                if range_pair[0] >= range_pair[1]:
                    continue

                fake_param = torch.nn.Parameter(torch.empty([0]))
                self.param_to_chunk32[fake_param] = chunk16.paired_chunk
                self.param_to_range[fake_param] = range_pair

                fake_params_list.append(fake_param)

            group['params'] = fake_params_list

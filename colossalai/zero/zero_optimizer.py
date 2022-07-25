import torch
import torch.distributed as dist
from enum import Enum
from torch.optim import Optimizer
from colossalai.nn.parallel.data_parallel import ZeroDDP
from typing import Dict
from colossalai.amp.naive_amp.grad_scaler import DynamicGradScaler
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.utils import get_current_device, disposable


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
                 max_scale: float = 2**32):
        super().__init__(optim)
        assert isinstance(module, ZeroDDP)
        self.module = module
        self.gemini_manager = module.gemini_manager
        self.chunk_manager = self.gemini_manager.chunk_manager
        self.optim_state = OptimState.UNSCALED
        self.fp16_param_to_fp32_param: Dict[torch.Tensor, torch.Tensor] = {}
        for p, fp32_p in zip(module.parameters(), module.fp32_params):
            self.fp16_param_to_fp32_param[p] = fp32_p

        # Grad scaler
        self.grad_scaler = DynamicGradScaler(initial_scale=initial_scale,
                                             min_scale=min_scale,
                                             growth_factor=growth_factor,
                                             backoff_factor=backoff_factor,
                                             growth_interval=growth_interval,
                                             hysteresis=hysteresis,
                                             max_scale=max_scale)
        self._found_overflow: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())
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

    def _update_params_ptr(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                if not self.module.chunk_manager.get_chunk(p).is_empty:
                    p.data = self.fp16_param_to_fp32_param[p]
                else:
                    assert p.grad is None

    def _update_fp16_params(self):
        self.module.chunk_manager.copy_chunk_group('fp16_param', 'fp32_param')

    def _check_overflow(self):
        # clear previous overflow record
        self._found_overflow.fill_(self.module.overflow_counter)

        # all-reduce across global group
        dist.all_reduce(self._found_overflow)

        return self._found_overflow.item() > 0

    def _unscale_grads(self):
        assert self.optim_state == OptimState.SCALED
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)
        self.optim_state = OptimState.UNSCALED

    @property
    def loss_scale(self):
        return self.grad_scaler.scale.item()

    def zero_grad(self, *args, **kwargs):
        self.module.overflow_counter = 0
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        self._maybe_move_fp32_params()
        # unscale grads if scaled
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        found_inf = self._check_overflow()
        self.grad_scaler.update(found_inf)
        if found_inf:
            self._logger.info(f'Found overflow. Skip step')
            self.zero_grad()
            self._update_fp16_params()
            return
        self._update_params_ptr()
        ret = self.optim.step(*args, **kwargs)
        self._register_states()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float):
        if self.optim_state == OptimState.SCALED:
            self._unscale_grads()
        # TODO(ver217): fix zero clip grad norm
        return super().clip_grad_norm(model, max_norm)

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
            for fp16_param_chunk, fp32_param_chunk in zip(self.chunk_manager.chunk_groups['fp16_param'],
                                                          self.chunk_manager.chunk_groups['fp32_param']):
                if fp32_param_chunk.is_empty:
                    continue
                if fp32_params_used_cuda_margin_mem + fp32_param_chunk.mem < fp32_params_available_cuda_margin_mem:
                    self.chunk_manager.move_chunk(fp32_param_chunk, get_current_device())
                    # stores grad now
                    self.chunk_manager.move_chunk(fp16_param_chunk, get_current_device())
                    self.module._set_chunk_grad_device(fp16_param_chunk, get_current_device())
                    fp32_params_used_cuda_margin_mem += fp32_param_chunk.mem
                    for p in fp16_param_chunk.get_tensors():
                        state = self.optim.state[p]
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(get_current_device())

            self.module._setup_grads_ptr()

    def _register_states_(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                state = self.optim.state[p]
                for val in state.values():
                    if isinstance(val, torch.Tensor):
                        self.chunk_manager.add_extern_static_tensor(val)

    def state_dict(self):
        optim_state_dict = super().state_dict()
        scaler_state_dict = self.grad_scaler.state_dict()
        optim_state_dict['scaler'] = scaler_state_dict
        return optim_state_dict

    def load_state_dict(self, *args, **kwargs):
        if 'scaler' not in args[0]:
            self._logger.warning('Missing scaler when loading optimizer state dict', ranks=[0])
        else:
            scaler_state_dict = args[0].pop('scaler')
            self.grad_scaler.load_state_dict(scaler_state_dict)
        super().load_state_dict(*args, **kwargs)
        for group in self.optim.param_groups:
            for p in group['params']:
                state = self.optim.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(dtype=self.fp16_param_to_fp32_param[p].dtype,
                                        device=self.fp16_param_to_fp32_param[p].device)

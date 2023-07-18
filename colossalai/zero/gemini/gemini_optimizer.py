# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import copy
import gc
import math
import warnings
from typing import Any, Dict, Set, Tuple

import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import Optimizer

from colossalai.amp.naive_amp.mixed_precision_mixin import BF16MixedPrecisionMixin, FP16MixedPrecisionMixin
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import ColossalaiOptimizer, CPUAdam, FusedAdam, HybridAdam
from colossalai.utils import disposable, get_current_device, is_ddp_ignored

from .chunk import Chunk, ChunkManager
from .gemini_ddp import ZeroDDP

__all__ = ['ZeroOptimizer', 'GeminiAdamOptimizer']

_AVAIL_OPTIM_LIST = {FusedAdam, CPUAdam, HybridAdam}


class GeminiFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):

    def __init__(self,
                 module: ZeroDDP,
                 initial_scale: float = 2**16,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32) -> None:
        super().__init__(initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis,
                         max_scale)
        self.module = module

    def check_local_overflow(self) -> bool:
        return self.module.overflow_counter > 0

    def pre_zero_grad(self) -> None:
        self.module.overflow_counter = 0


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
        growth_factor (float, optional): Growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): Backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): Growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): Hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): Max_scale used by DynamicGradScaler. Defaults to 2**32.
        clipping_norm (float, optional): The norm value used to clip gradient. Defaults to 0.0.
        norm_type (float, optional): The type of norm used for gradient clipping. Currently, only L2-norm (norm_type=2.0)
            is supported in ZeroOptimizer. Defaults to 2.0.
        verbose (bool, optional): Whether to print verbose information, including grad overflow info. Defaults to False.
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
                 verbose: bool = False,
                 **defaults: Any):
        super().__init__(optim)
        assert isinstance(module, ZeroDDP)
        assert type(optim) in _AVAIL_OPTIM_LIST, "You should use an optimizer in the available list:\n" \
            f"{_AVAIL_OPTIM_LIST}"
        self.module = module
        self.gemini_manager = module.gemini_manager
        self.chunk_manager: ChunkManager = self.gemini_manager.chunk_manager
        self.param_to_range: Dict[Parameter, Tuple[int, int]] = dict()
        self.param_to_chunk32: Dict[Parameter, Chunk] = dict()
        self.chunk16_set: Set[Chunk] = set()
        self.clipping_flag = clipping_norm > 0.0
        self.max_norm = clipping_norm
        self.verbose = verbose
        self.param_groups_backup = list()

        # Mapping from integer id to real/fake param tensor, used for checkpointing.
        self.id_to_real_params: Dict[int, Parameter] = dict()
        self.id_to_fake_params: Dict[int, Parameter] = dict()

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

        if module.mixed_precision is torch.float16:
            self.mix_precision_mixin = GeminiFP16MixedPrecisionMixin(module,
                                                                     initial_scale=initial_scale,
                                                                     min_scale=min_scale,
                                                                     growth_factor=growth_factor,
                                                                     backoff_factor=backoff_factor,
                                                                     growth_interval=growth_interval,
                                                                     hysteresis=hysteresis,
                                                                     max_scale=max_scale)
        elif module.mixed_precision is torch.bfloat16:
            self.mix_precision_mixin = BF16MixedPrecisionMixin()
        else:
            raise RuntimeError(f"Unsupported mixed precision type: {module.mixed_precision}")

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
                fake_param.data = none_tensor.to(fake_param.device)

        for chunk16 in self.chunk16_set:
            chunk16.optim_update()

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
        div_scale = self.mix_precision_mixin.get_grad_div_scale()

        if self.clipping_flag:
            total_norm = self._calc_global_norm()
            clip = ((total_norm / div_scale) + 1e-6) / self.max_norm
            if clip > 1:
                div_scale = clip * div_scale

        return -1 if div_scale == 1.0 else div_scale

    def zero_grad(self, *args, **kwargs):
        self.mix_precision_mixin.pre_zero_grad()
        return self.optim.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        self._maybe_move_fp32_params()
        self._set_grad_ptr()

        if self.mix_precision_mixin.should_skip_step():
            if self.verbose:
                self._logger.info(f'Found overflow. Skip step')
            self._clear_global_norm()    # clear recorded norm
            self.zero_grad()    # reset all gradients
            self._update_fp16_params()
            return

        # get combined scale. combined scale = loss scale * clipping norm
        # so that gradient = gradient / combined scale
        combined_scale = self._get_combined_scale()

        ret = self.optim.step(div_scale=combined_scale, *args, **kwargs)
        self._register_states()
        self.zero_grad()
        self._update_fp16_params()
        return ret

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float, norm_type: float = 2.0):
        raise NotImplementedError

    def backward(self, loss: torch.Tensor):
        loss = self.mix_precision_mixin.pre_backward(loss)
        self.module.backward(loss)

    def backward_by_grad(self, tensor: torch.Tensor, grad: torch.Tensor):
        # This function is called except the last stage of pipeline parallel
        # It receives the scaled grad from the previous rank
        # No need to scale the grad again
        # Need to unscale when optimizing
        grad = self.mix_precision_mixin.pre_backward_by_grad(grad)
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

        param_id = -1
        for group in self.optim.param_groups:
            fake_params_list = list()
            group_backup = {k: v for k, v in group.items() if k != 'params'}
            group_ids = []
            for param in group['params']:

                # Record the mapping of id to current param.
                param_id += 1
                self.id_to_real_params[param_id] = param
                group_ids.append(param_id)

                # If current param is controlled by current process, add it to fake_param.
                if is_ddp_ignored(param):
                    continue
                chunk16 = self.chunk_manager.get_chunk(param)
                range_pair = get_range_pair(chunk16, param)
                if range_pair[0] >= range_pair[1]:
                    continue
                grad_device = self.module.grads_device[param]
                fake_param = torch.nn.Parameter(torch.empty([0], device=grad_device))
                self.param_to_chunk32[fake_param] = chunk16.paired_chunk
                self.param_to_range[fake_param] = range_pair
                self.id_to_fake_params[param_id] = fake_param
                fake_params_list.append(fake_param)

            # Update self.optim.param_groups as well as backup group.
            group['params'] = fake_params_list
            group_backup['params'] = group_ids
            self.param_groups_backup.append(group_backup)

    def get_offsets(self, param_id: int) -> tuple:
        '''
        Args:
            param_id(int): The id of parameter.

        Returns:
            chunk_offset(int): Offset of parameter inside the chunk.
            shard_offset(int): Offset of its optimizer state shard
                                relative to the whole optimizer state.
            shard_size(int): Length of parameter shard owned by current process.
        '''

        if param_id not in self.id_to_fake_params:
            return -1, -1, -1
        fake_param = self.id_to_fake_params[param_id]
        chunk = self.param_to_chunk32[fake_param].paired_chunk
        param = self.id_to_real_params[param_id]
        param_info = chunk.tensors_info[param]

        begin_in_chunk, end_in_chunk = self.param_to_range[fake_param]
        chunk_offset = begin_in_chunk
        shard_offset = begin_in_chunk + chunk.shard_begin - param_info.offset
        shard_size = end_in_chunk - begin_in_chunk
        assert chunk_offset >= 0 and shard_offset >= 0

        return chunk_offset, shard_offset, shard_size

    def collect_states(self, param_id: int, only_rank_0: bool = True) -> dict:
        """
        Args:
            param_id (int): id of the parameter whose state is to be gathered at master rank.
            only_rank_0(bool): if True, states will be collected only on master rank, otherwise collected on every rank.

        Returns:
            collected_states(dict): the gathered optimzier state of parameter with given id
                                    if this method is called by master rank, otherwise an empty dict.

        This method can work only when called by all processes simultaneously.
        """

        # Get param & chunk & process group.
        param = self.id_to_real_params[param_id]
        fake_param = self.id_to_fake_params.get(param_id, None)
        chunk = self.chunk_manager.get_chunk(param)
        process_group = chunk.torch_pg
        rank = dist.get_rank(process_group)
        master_rank = 0
        collected_states = {}

        # Fetch names of states through all_gather.
        local_state_names = None
        if fake_param is not None:
            local_state_names = list(self.optim.state[fake_param].keys())
        gathered_state_names = [None for _ in range(dist.get_world_size(process_group))]
        dist.barrier()
        dist.all_gather_object(gathered_state_names, local_state_names)
        state_names = None
        for names in gathered_state_names:
            if names is not None:
                # Assume different devices share the same set of state names if they have.
                state_names = copy.deepcopy(names)
                break

        # Directly return if this parameter doesn't have optimizer states.
        # e.g. parameter freezed/layer dropped
        if state_names is None:
            return collected_states

        # Boolean variable is_collector indicates that whether the current rank
        # needs to gather the whole optimizer states.
        # Only master rank is collector when only_rank_0 is True.
        # Every rank is collector when only_rank_0 is False.
        is_collector = (rank == master_rank) or (not only_rank_0)

        # If the chunk is kept gathered,
        # the parameteres are treated the same as that of those in strict DDP during training.
        # So states can be directly fetched from current device.
        if chunk.keep_gathered:
            assert param_id in self.id_to_fake_params
            if is_collector:
                states = self.optim.state[fake_param]
                for state_name in state_names:
                    if state_name == 'step':
                        # To keep aligned with pytorch, state 'step' is stored as a pytorch tensor with type float32.
                        collected_states[state_name] = torch.tensor(states['step'],
                                                                    dtype=torch.float32,
                                                                    requires_grad=False).cpu()
                    else:
                        collected_states[state_name] = states[state_name].detach().clone().to(torch.float32).cpu()
            return collected_states

        # Check whether the param with given id is managed by current process.
        own_param = param_id in self.id_to_fake_params

        # Collector gets prepared for state collecting.
        if is_collector:
            for state_name in state_names:
                if state_name == 'step':
                    # To keep aligned with pytorch, state 'step' is stored as a pytorch tensor with type float32.
                    collected_states[state_name] = torch.tensor(0.0, dtype=torch.float32, requires_grad=False).cpu()
                else:
                    collected_states[state_name] = torch.zeros(param.numel(), dtype=torch.float32,
                                                               requires_grad=False).cpu()

        # Materials for gathering, including compacted state tensors, and the offset of shard inside each state.
        compacted_states = self.pack_optimizer_states_to_tensor(param_id, state_names) if own_param else None
        _, shard_offset, shard_size = self.get_offsets(param_id)

        # Collectors gather state shards through all_gathering.
        gathered_state_shards = [None for _ in range(dist.get_world_size(process_group))]

        dist.barrier()
        dist.all_gather_object(gathered_state_shards, [compacted_states, shard_offset, shard_size])

        if is_collector:
            for state_shard in gathered_state_shards:
                compacted_states = state_shard[0]
                shard_offset = state_shard[1]
                shard_size = state_shard[2]
                if compacted_states is None:
                    continue
                self.load_from_compacted_states(compacted_states, collected_states, state_names, shard_offset,
                                                shard_size)

        # Clean gathered states
        for state_shard in gathered_state_shards:
            del state_shard[0]
            gc.collect()

        # Reshape tensors
        if is_collector:
            for state_name, state_tensor in collected_states.items():
                if state_tensor.numel() == param.numel():
                    collected_states[state_name] = torch.reshape(state_tensor, param.shape)

        return collected_states

    def pack_optimizer_states_to_tensor(self,
                                        param_id: int,
                                        state_names: list,
                                        device: torch.device = torch.device('cuda'),
                                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
        '''
        With param id given, pack its optimizer states into a compact tensor and return.
        '''
        if param_id not in self.id_to_fake_params:
            return None

        fake_param = self.id_to_fake_params[param_id]
        param_range = self.param_to_range[fake_param]
        states = self.optim.state[fake_param]
        shard_size = param_range[1] - param_range[0]
        compacted_size = 0
        for name in state_names:
            if name == 'step':
                compacted_size += 1
            else:
                compacted_size += shard_size
        compacted_states = torch.zeros(compacted_size, dtype=dtype, device=device, requires_grad=False)

        next_state_offset = 0
        for state_name, state_tensor in states.items():
            # State 'step' needs special operation.
            if state_name == 'step':
                if isinstance(state_tensor, torch.Tensor):
                    compacted_states[next_state_offset] = state_tensor[0].item()
                else:
                    assert isinstance(state_tensor, int)
                    compacted_states[next_state_offset] = state_tensor
                next_state_offset += 1
            else:
                assert state_tensor.numel() == shard_size
                compacted_states[next_state_offset:next_state_offset + shard_size].copy_(state_tensor)
                next_state_offset += shard_size

        return compacted_states

    def load_from_compacted_states(self, compacted_states: torch.Tensor, collected_states: dict, state_names: list,
                                   shard_start: int, shard_size: int):
        '''
        Given a tensor carrying compacted optimizer states,
        update these states to collected_states.
        '''
        shard_end = shard_start + shard_size
        next_state_offset = 0

        for state_name in state_names:
            if state_name == 'step':
                collected_states['step'].data = torch.tensor(compacted_states[next_state_offset].item(),
                                                             dtype=torch.float32,
                                                             requires_grad=False).cpu()
                next_state_offset += 1
            else:
                target_segment = collected_states[state_name][shard_start:shard_end]
                target_segment.copy_(compacted_states[next_state_offset:next_state_offset + shard_size])
                next_state_offset += shard_size

    def state_dict(self, only_rank_0: bool = True) -> dict:
        """
        Args:
            only_rank_0 (bool): a boolean value indicating whether the state_dict is collected
            only on rank 0, dafault to True.

        Returns:
            The complete state of the optimizer as a :class:`dict`.
            It contains two entries:

            * state - a dict holding current optimization state. Its content
                differs between optimizer classes.
            * param_groups - a list containing all parameter groups where each
                parameter group is a dict.

        Warning: This method will gather and return the whole optimizer state_dict,
                 so it should be called only when memory resources are abundant.
        """
        state_dict = {}
        state_dict['param_groups'] = copy.deepcopy(self.param_groups_backup)

        torch_special_hyperparameters = {
            'amsgrad': False,
            'maximize': False,
            'foreach': None,
            'capturable': False,
            'differentiable': False,
            'fused': False
        }

        for group in state_dict['param_groups']:
            for k, v in torch_special_hyperparameters.items():
                if k not in group:
                    group[k] = v

        # Collect optimizer states.
        state_dict['state'] = dict()
        for param_id in self.id_to_real_params.keys():
            dist.barrier()
            state_dict['state'][param_id] = self.collect_states(param_id=param_id, only_rank_0=only_rank_0)
        return state_dict

    def load_param_groups(self, saved_param_groups: list):
        """
        Load saved_param_groups into
        self.param_groups and self.param_groups_backup
        """
        self.param_groups_backup = copy.deepcopy(saved_param_groups)

        # discard the older param_groups
        self.optim.param_groups = []

        for group in saved_param_groups:
            fake_params_list = list()
            updated_group = {k: v for k, v in group.items() if k != 'params'}
            for param_id in group['params']:
                if param_id not in self.id_to_fake_params:
                    continue
                fake_param = self.id_to_fake_params[param_id]
                fake_params_list.append(fake_param)
            updated_group['params'] = fake_params_list
            self.optim.param_groups.append(updated_group)

    def load_single_param_states(self, param_id: int, saved_states: dict):
        """
        Load saved optimizer states into parameter with given id.
        """

        def cast(param, state_range, value, key=None):
            """
            Make a copy of the needed segment of value and cast it to device of param.
            """
            assert isinstance(value, torch.Tensor)
            ret_val = value
            if (key == "step"):
                assert value.numel() == 1
                ret_val = int(value.item())
            else:
                state_start, state_end = state_range
                ret_val = torch.zeros(state_end - state_start,
                                      dtype=torch.float32,
                                      device=param.device,
                                      requires_grad=False)
                ret_val.copy_(value.flatten()[state_start:state_end])
            return ret_val

        assert param_id in self.id_to_fake_params
        fake_param = self.id_to_fake_params[param_id]
        _, state_offset, param_size = self.get_offsets(param_id)
        state_range = (state_offset, state_offset + param_size)

        # Copy states assigned to param (and cast tensors to appropriate types).
        updated_states = dict()
        for k, v in saved_states.items():
            updated_states[k] = cast(fake_param, state_range, v, k)
            del v    # clean loaded states
        self.optim.state[fake_param].update(updated_states)

    def load_state_dict(self, state_dict: dict):
        """Loads optimizer state from whole optimizer state_dict.
           During loading, filter out the part of states not considered by current process.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        assert 'param_groups' in state_dict
        self.load_param_groups(state_dict['param_groups'])

        state = state_dict['state']

        for param_id, param_states in state.items():
            if param_id in self.id_to_fake_params:
                self.load_single_param_states(param_id, param_states)

        # Epilogue for pytorch optimizer.
        self.optim._hook_for_profile()    # To support multiprocessing pickle/unpickle.
        self.optim.defaults.setdefault('differentiable', False)


class GeminiAdamOptimizer(ZeroOptimizer):

    def __init__(self, model: torch.nn.Module, **defaults: Any) -> None:
        optimizer = HybridAdam(model.parameters(), **defaults)
        super().__init__(optimizer, model, **defaults)

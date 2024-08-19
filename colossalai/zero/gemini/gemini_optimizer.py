# this code is inspired by the DeepSpeed library and implemented with our own design from scratch
import copy
import math
from typing import Any, Dict, Iterator, OrderedDict, Set, Tuple, Union

import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed import ProcessGroup
from torch.nn import Parameter
from torch.optim import Optimizer

from colossalai.accelerator import get_accelerator
from colossalai.amp.naive_amp.mixed_precision_mixin import BF16MixedPrecisionMixin, FP16MixedPrecisionMixin
from colossalai.checkpoint_io.utils import StateDictSharder, gather_distributed_param
from colossalai.interface import OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import CPUAdam, FusedAdam, HybridAdam
from colossalai.tensor.d_tensor import (
    distribute_tensor,
    distribute_tensor_with_customization,
    get_device_mesh,
    get_global_shape,
    get_sharding_spec,
    init_as_dtensor,
    init_tensor_as_customization_distributed,
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from colossalai.tensor.padded_tensor import (
    init_as_padded_tensor,
    is_padded_tensor,
    to_padded_tensor,
    to_unpadded_tensor,
)
from colossalai.utils import disposable, is_ddp_ignored

from .chunk import Chunk, ChunkManager
from .gemini_ddp import GeminiDDP

__all__ = ["GeminiOptimizer", "GeminiAdamOptimizer"]

_AVAIL_OPTIM_LIST = {FusedAdam, CPUAdam, HybridAdam}


class GeminiFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):
    def __init__(
        self,
        module: GeminiDDP,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        super().__init__(
            initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis, max_scale
        )
        self.module = module

    def check_local_overflow(self) -> bool:
        return self.module.chunk_manager.overflow_counter.item() > 0

    def pre_zero_grad(self) -> None:
        self.module.chunk_manager.overflow_counter.zero_()


class GeminiOptimizer(OptimizerWrapper):
    """A wrapper for optimizer. ``GeminiDDP`` and ``GeminiOptimizer`` implement Zero Redundancy Optimizer (ZeRO state-3).

    Note:
        You must use ``GeminiDDP`` with ``GeminiOptimizer``.

    Note:
        Make sure you set ``placement_policy`` of ``GeminiManager`` to `"auto"`,
        if you set ``gpu_margin_mem_ratio > 0``.

    Args:
        optim (Optimizer): An Optimizer instance.
        module (GeminiDDP): A ``GeminiDDP`` instance.
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
        max_norm (float, optional): The norm value used to clip gradient. Defaults to 0.0.
        norm_type (float, optional): The type of norm used for gradient clipping. Currently, only L2-norm (norm_type=2.0)
            is supported in GeminiOptimizer. Defaults to 2.0.
        verbose (bool, optional): Whether to print verbose information, including grad overflow info. Defaults to False.
    """

    def __init__(
        self,
        optim: Optimizer,
        module: GeminiDDP,
        gpu_margin_mem_ratio: float = 0.0,
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
        tp_group: ProcessGroup = None,
        params_info=None,
        verbose: bool = False,
        **defaults: Any,
    ):
        super().__init__(optim)
        assert isinstance(module, GeminiDDP)
        assert type(optim) in _AVAIL_OPTIM_LIST, (
            "You should use an optimizer in the available list:\n" f"{_AVAIL_OPTIM_LIST}"
        )
        self.module = module
        self.gemini_manager = module.gemini_manager
        self.chunk_manager: ChunkManager = self.gemini_manager.chunk_manager
        self.param_to_range: Dict[Parameter, Tuple[int, int]] = dict()
        self.param_to_chunk16: Dict[Parameter, Chunk] = dict()
        self.chunk16_set: Set[Chunk] = set()
        self.clipping_flag = max_norm > 0.0
        self.max_norm = max_norm
        self.tp_group = tp_group
        self.params_info = params_info
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.verbose = verbose
        self.param_groups_backup = list()
        self.logger = get_dist_logger()
        # Mapping from integer id to real/fake param tensor, used for checkpointing.
        self.id_to_real_params: Dict[int, Parameter] = dict()
        self.id_to_fake_params: Dict[int, Parameter] = dict()

        if self.clipping_flag:
            assert norm_type == 2.0, "GeminiOptimizer only supports L2 norm now"

        ddp_param_list = []
        for name, param in module.named_parameters():
            if is_ddp_ignored(param):
                if param.requires_grad:
                    self.logger.warning(
                        f"Parameter `{name}` is ignored by DDP but requires gradient! "
                        "You should handle its optimizer update by yourself!",
                        ranks=[0],
                    )
            else:
                ddp_param_list.append(param)

        for p in ddp_param_list:
            chunk_16 = self.chunk_manager.get_chunk(p)
            if chunk_16 not in self.chunk16_set:
                chunk_16.l2_norm_flag = self.clipping_flag
                self.chunk16_set.add(chunk_16)

        self.__init__optimizer()

        if module.mixed_precision is torch.float16:
            self.mix_precision_mixin = GeminiFP16MixedPrecisionMixin(
                module,
                initial_scale=initial_scale,
                min_scale=min_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                hysteresis=hysteresis,
                max_scale=max_scale,
            )
        elif module.mixed_precision is torch.bfloat16:
            self.mix_precision_mixin = BF16MixedPrecisionMixin()
        else:
            raise RuntimeError(f"Unsupported mixed precision type: {module.mixed_precision}")

        self._logger = get_dist_logger()

        self.gpu_margin_mem_ratio: float = float(gpu_margin_mem_ratio)
        assert 0.0 <= self.gpu_margin_mem_ratio <= 1.0, f"gpu_margin_mem_ratio must >=0.0 and <=1.0"
        # Only move fp32 shards from CPU to GPU when user allows and inner optimizer is valid
        # Inner optimizer must support optimizing hybrid (CPU and CUDA) tensors,
        # and it must set `num_fp32_shards_per_param` correctly
        self._should_move_fp32_params_h2d: bool = (
            self.gemini_manager.is_cuda_margin_mem_avail
            and self.gpu_margin_mem_ratio > 0.0
            and getattr(optim, "num_fp32_shards_per_param", 0) >= 2
        )
        if self.gpu_margin_mem_ratio > 0.0 and not self.gemini_manager.is_cuda_margin_mem_avail:
            self._logger.warning(f'gpu_margin_mem_ratio is meaningless when placement_policy is not "auto"', ranks=[0])

        self._register_states = disposable(self._register_states_)

    def _set_grad_ptr(self):
        for group in self.param_groups:
            for fake_param in group["params"]:
                chunk16 = self.param_to_chunk16[fake_param]
                begin, end = self.param_to_range[fake_param]

                grad_chunk16 = chunk16 if self.module.chunk_manager.reuse_fp16_chunk else chunk16.grad_chunk
                fake_param.data = grad_chunk16.payload[begin:end]
                fake_param.grad = fake_param.data

                to_update_chunk = chunk16.paired_chunk if self.module.master_weights else chunk16
                fake_param.data = to_update_chunk.payload[begin:end]

    def _update_fp16_params(self):
        none_tensor = torch.empty([0])
        for group in self.param_groups:
            for fake_param in group["params"]:
                assert fake_param.grad is None
                fake_param.data = none_tensor.to(fake_param.device)

        for chunk16 in self.chunk16_set:
            chunk16.optim_update()

    def _clear_global_norm(self) -> None:
        for c16 in self.chunk16_set:
            grad_chunk = c16 if self.module.chunk_manager.reuse_fp16_chunk else c16.grad_chunk
            grad_chunk.l2_norm = None

    def _calc_global_norm(self) -> float:
        norm_sqr: float = 0.0
        group_to_norm = dict()
        for c16 in self.chunk16_set:
            grad_chunk = c16 if self.module.chunk_manager.reuse_fp16_chunk else c16.grad_chunk
            assert grad_chunk.l2_norm is not None

            if grad_chunk.is_gathered:
                norm_sqr += grad_chunk.l2_norm
            else:
                # this chunk is sharded, use communication to collect total norm
                if grad_chunk.torch_pg not in group_to_norm:
                    group_to_norm[grad_chunk.torch_pg] = 0.0
                group_to_norm[grad_chunk.torch_pg] += grad_chunk.l2_norm

            grad_chunk.l2_norm = None  # clear l2 norm

        comm_buffer = torch.zeros(1, dtype=torch.float, device=get_accelerator().get_current_device())
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
        if self.module.master_weights:
            self._maybe_move_fp32_params()
        self._set_grad_ptr()

        if self.mix_precision_mixin.should_skip_step():
            if self.verbose:
                self._logger.info(f"Found overflow. Skip step")
            self._clear_global_norm()  # clear recorded norm
            self.zero_grad()  # reset all gradients
            if self.module.chunk_manager.reuse_fp16_chunk:
                self._update_fp16_params()
            return

        # get combined scale. combined scale = loss scale * clipping norm
        # so that gradient = gradient / combined scale
        combined_scale = self._get_combined_scale()

        ret = self.optim.step(div_scale=combined_scale, *args, **kwargs)
        self._register_states()
        self.zero_grad()
        if self.module.master_weights:
            self._update_fp16_params()
        self.module.chunk_manager.accumulating_grads = False
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
                for fake_param in group["params"]:
                    chunk16 = self.param_to_chunk16[fake_param]
                    chunk32 = chunk16.paired_chunk

                    if chunk32.device_type == "cuda" or chunk32.device_type == "npu":
                        continue

                    if fp32_params_used_cuda_margin_mem + chunk32.payload_mem < fp32_params_available_cuda_margin_mem:
                        self.chunk_manager.move_chunk(chunk32, get_accelerator().get_current_device())
                        # stores grad now
                        self.chunk_manager.move_chunk(chunk16, get_accelerator().get_current_device())
                        self.module.set_chunk_grad_device(chunk16, get_accelerator().get_current_device())
                        fp32_params_used_cuda_margin_mem += chunk32.payload_mem

            for group in self.param_groups:
                for fake_param in group["params"]:
                    chunk16 = self.param_to_chunk16[fake_param]
                    chunk32 = chunk16.paired_chunk
                    if chunk32.device_type == "cuda" or chunk32.device_type == "npu":
                        state = self.optim.state[fake_param]
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(get_accelerator().get_current_device())

    def _register_states_(self):
        for group in self.optim.param_groups:
            for p in group["params"]:
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
            group_backup = {k: v for k, v in group.items() if k != "params"}
            group_ids = []
            for param in group["params"]:
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
                self.param_to_chunk16[fake_param] = chunk16
                self.param_to_range[fake_param] = range_pair
                self.id_to_fake_params[param_id] = fake_param
                fake_params_list.append(fake_param)

            # Update self.optim.param_groups as well as backup group.
            group["params"] = fake_params_list
            group_backup["params"] = group_ids
            self.param_groups_backup.append(group_backup)

    def get_offsets(self, param_id: int) -> tuple:
        """
        Args:
            param_id(int): The id of parameter.

        Returns:
            chunk_offset(int): Offset of parameter inside the chunk.
            shard_offset(int): Offset of its optimizer state shard
                                relative to the whole optimizer state.
            shard_size(int): Length of parameter shard owned by current process.
        """

        if param_id not in self.id_to_fake_params:
            return -1, -1, -1
        fake_param = self.id_to_fake_params[param_id]
        chunk = self.param_to_chunk16[fake_param]
        param = self.id_to_real_params[param_id]
        param_info = chunk.tensors_info[param]

        begin_in_chunk, end_in_chunk = self.param_to_range[fake_param]
        chunk_offset = begin_in_chunk
        if chunk.keep_gathered:
            shard_offset = 0
        else:
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
            collected_states(dict): the gathered optimizer state of parameter with given id
                                    if this method is called by master rank, otherwise an empty dict.

        This method can work only when called by all processes simultaneously.
        """

        # Get param & chunk & process group.
        param = self.id_to_real_params[param_id]
        fake_param = self.id_to_fake_params.get(param_id, None)
        chunk = self.chunk_manager.get_chunk(param)
        zero_group = chunk.torch_pg
        rank = dist.get_rank(zero_group)
        master_rank = 0
        collected_states = {}

        # Fetch names of states through all_gather.
        local_state_names = None
        if fake_param is not None:
            local_state_names = list(self.optim.state[fake_param].keys())
        gathered_state_names = [None for _ in range(dist.get_world_size(zero_group))]
        dist.barrier()
        dist.all_gather_object(gathered_state_names, local_state_names, zero_group)
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

        # get tensor parallelism information
        is_dtensor = is_distributed_tensor(param)
        is_customized_distributed = is_customized_distributed_tensor(param)
        shard_spec = get_sharding_spec(param) if is_dtensor else None
        device_mesh = get_device_mesh(param) if is_dtensor else None
        global_shape = self.params_info["id2shape"][param_id]

        # If the chunk is kept gathered,
        # the parameters are treated the same as that of those in strict DDP during training.
        # So states can be directly fetched from current device.
        if chunk.keep_gathered:
            assert param_id in self.id_to_fake_params
            if is_collector:
                states = self.optim.state[fake_param]
                for state_name in state_names:
                    if state_name == "step":
                        # To keep aligned with pytorch, state 'step' is stored as a pytorch tensor with type float32.
                        collected_states[state_name] = torch.tensor(
                            states["step"], dtype=torch.float32, requires_grad=False
                        ).cpu()
                    else:
                        state_tensor = states[state_name].detach().clone().to(torch.float32).cpu()
                        if is_dtensor:
                            global_shape = get_global_shape(param)
                            state_tensor = torch.reshape(state_tensor, param.shape).to(param.device)
                            state_tensor = init_as_dtensor(
                                state_tensor,
                                device_mesh=device_mesh,
                                sharding_spec=shard_spec,
                                global_shape=global_shape,
                            )
                        elif is_customized_distributed:
                            state_tensor = torch.reshape(state_tensor, param.shape).to(param.device)
                            init_tensor_as_customization_distributed(
                                state_tensor, shard_fn=param.shard_fn, gather_fn=param.gather_fn
                            )
                        state_tensor = gather_distributed_param(state_tensor, keep_vars=False).cpu()
                        state_tensor = state_tensor.reshape(global_shape)
                        if is_padded_tensor(param):
                            state_tensor = init_as_padded_tensor(
                                state_tensor, param._current_length, param._origin_length, param._padding_dim
                            )
                            state_tensor = to_unpadded_tensor(state_tensor)
                        collected_states[state_name] = state_tensor
            return collected_states

        # Check whether the param with given id is managed by current process.
        own_param = param_id in self.id_to_fake_params

        # Collector gets prepared for state collecting.
        if is_collector:
            for state_name in state_names:
                if state_name == "step":
                    # To keep aligned with pytorch, state 'step' is stored as a pytorch tensor with type float32.
                    collected_states[state_name] = torch.tensor(0.0, dtype=torch.float32, requires_grad=False).cpu()
                else:
                    collected_states[state_name] = torch.zeros(
                        param.numel(), dtype=torch.float32, requires_grad=False
                    ).cpu()

        # Materials for gathering, including compacted state tensors, and the offset of shard inside each state.
        compacted_states = self.pack_optimizer_states_to_tensor(param_id, state_names) if own_param else None
        _, shard_offset, shard_size = self.get_offsets(param_id)

        # Collectors gather state shards through all_gathering.
        gathered_state_shards = [None for _ in range(dist.get_world_size(zero_group))]

        dist.barrier()
        dist.all_gather_object(gathered_state_shards, [compacted_states, shard_offset, shard_size], group=zero_group)

        if is_collector:
            for state_shard in gathered_state_shards:
                compacted_states = state_shard[0]
                shard_offset = state_shard[1]
                shard_size = state_shard[2]
                if compacted_states is None:
                    continue
                self.load_from_compacted_states(
                    compacted_states, collected_states, state_names, shard_offset, shard_size
                )

        # Reshape tensors
        if is_collector:
            for state_name, state_tensor in collected_states.items():
                if state_tensor.numel() == param.numel():
                    collected_states[state_name] = torch.reshape(state_tensor, param.shape)
                if is_dtensor:
                    global_shape = get_global_shape(param)
                    state_tensor = state_tensor.to(param.device)
                    state_tensor = init_as_dtensor(
                        state_tensor, sharding_spec=shard_spec, device_mesh=device_mesh, global_shape=global_shape
                    )
                elif is_customized_distributed:
                    state_tensor = state_tensor.to(param.device)
                    init_tensor_as_customization_distributed(
                        state_tensor, shard_fn=param.shard_fn, gather_fn=param.gather_fn
                    )
                state_tensor = gather_distributed_param(state_tensor, keep_vars=False).cpu()
                if is_padded_tensor(param):
                    state_tensor = init_as_padded_tensor(
                        state_tensor, param._current_length, param._origin_length, param._padding_dim
                    )
                    state_tensor = to_unpadded_tensor(state_tensor)

        return collected_states

    def pack_optimizer_states_to_tensor(
        self,
        param_id: int,
        state_names: list,
        device: torch.device = get_accelerator().get_current_device(),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        With param id given, pack its optimizer states into a compact tensor and return.
        """
        if param_id not in self.id_to_fake_params:
            return None

        fake_param = self.id_to_fake_params[param_id]
        param_range = self.param_to_range[fake_param]
        states = self.optim.state[fake_param]
        shard_size = param_range[1] - param_range[0]
        compacted_size = 0
        for name in state_names:
            if name == "step":
                compacted_size += 1
            else:
                compacted_size += shard_size
        compacted_states = torch.zeros(compacted_size, dtype=dtype, device=device, requires_grad=False)

        next_state_offset = 0
        for state_name, state_tensor in states.items():
            # State 'step' needs special operation.
            if state_name == "step":
                if isinstance(state_tensor, torch.Tensor):
                    compacted_states[next_state_offset] = state_tensor[0].item()
                else:
                    assert isinstance(state_tensor, int)
                    compacted_states[next_state_offset] = state_tensor
                next_state_offset += 1
            else:
                assert state_tensor.numel() == shard_size
                compacted_states[next_state_offset : next_state_offset + shard_size].copy_(state_tensor)
                next_state_offset += shard_size

        return compacted_states

    def load_from_compacted_states(
        self,
        compacted_states: torch.Tensor,
        collected_states: dict,
        state_names: list,
        shard_start: int,
        shard_size: int,
    ):
        """
        Given a tensor carrying compacted optimizer states,
        update these states to collected_states.
        """
        shard_end = shard_start + shard_size
        next_state_offset = 0

        for state_name in state_names:
            if state_name == "step":
                collected_states["step"].data = torch.tensor(
                    compacted_states[next_state_offset].item(), dtype=torch.float32, requires_grad=False
                ).cpu()
                next_state_offset += 1
            else:
                target_segment = collected_states[state_name][shard_start:shard_end]
                target_segment.copy_(compacted_states[next_state_offset : next_state_offset + shard_size])
                next_state_offset += shard_size

    def get_param_groups_for_saving(self) -> list:
        """
        Return the param_groups in Pytorch format when saving to checkpoint.
        """

        param_groups = [
            {**group, "params": group_info["params"]}
            for group, group_info in zip(self.optim.param_groups, self.param_groups_backup)
        ]

        # To be compatible with pytorch checkpointing,
        # store extra hyperparameters used by pytorch Adam optimizer.
        torch_special_hyperparameters = {
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        }

        for group in param_groups:
            for k, v in torch_special_hyperparameters.items():
                if k not in group:
                    group[k] = v

        return param_groups

    def state_dict(self, only_rank_0: bool = True) -> dict:
        """
        Args:
            only_rank_0 (bool): a boolean value indicating whether the state_dict is collected
            only on rank 0, default to True.

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
        state_dict["param_groups"] = self.get_param_groups_for_saving()

        # Collect optimizer states.
        state_dict["state"] = dict()
        for param_id in self.id_to_real_params.keys():
            dist.barrier()
            state_dict["state"][param_id] = self.collect_states(param_id=param_id, only_rank_0=only_rank_0)
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
            updated_group = {k: v for k, v in group.items() if k != "params"}
            for param_id in group["params"]:
                if param_id not in self.id_to_fake_params:
                    continue
                fake_param = self.id_to_fake_params[param_id]
                fake_params_list.append(fake_param)
            updated_group["params"] = fake_params_list
            self.optim.param_groups.append(updated_group)

    def load_single_param_states(self, param_id: int, saved_states: dict):
        """
        Load saved optimizer states into parameter with given id.
        """

        def cast(param, state_range, value, global_shape, origin_shape, key=None):
            """
            Make a copy of the needed segment of value and cast it to device of param.
            """
            assert isinstance(value, torch.Tensor)
            ret_val = value
            if key == "step":
                assert value.numel() == 1
                ret_val = int(value.item())
            else:
                state_start, state_end = state_range
                ret_val = torch.zeros(
                    state_end - state_start, dtype=torch.float32, device=param.device, requires_grad=False
                )

                if is_dtensor:
                    global_shape = get_global_shape(real_param)

                if is_padded_tensor(real_param):
                    value = torch.reshape(value, origin_shape)
                    padding_dim = real_param._padding_dim
                    value = to_padded_tensor(value, global_shape[padding_dim], padding_dim)

                if is_dtensor:
                    value = distribute_tensor(value, sharding_spec=shard_spec, device_mesh=device_mesh)
                elif is_customized_distributed:
                    value = torch.reshape(value, global_shape)
                    value = distribute_tensor_with_customization(value, real_param.shard_fn, real_param.gather_fn)

                ret_val.copy_(value.flatten()[state_start:state_end])
            return ret_val

        assert param_id in self.id_to_fake_params
        fake_param = self.id_to_fake_params[param_id]
        _, state_offset, param_size = self.get_offsets(param_id)
        state_range = (state_offset, state_offset + param_size)

        # Copy states assigned to param (and cast tensors to appropriate types).
        updated_states = dict()

        # get tensor parallelism information
        real_param = self.id_to_real_params[param_id]
        is_dtensor = is_distributed_tensor(real_param)
        is_customized_distributed = is_customized_distributed_tensor(real_param)
        shard_spec = get_sharding_spec(real_param) if is_dtensor else None
        device_mesh = get_device_mesh(real_param) if is_dtensor else None
        global_shape = self.params_info["id2shape"][param_id]
        origin_shape = global_shape

        for k, v in saved_states.items():
            updated_states[k] = cast(fake_param, state_range, v, global_shape, origin_shape, k)
            del v  # clean loaded states
        self.optim.state[fake_param].update(updated_states)

    def load_param_states(self, param_states: dict):
        """Loads param states from a state_dict. The param_states can be complete or sharded.
           During loading, filter out the part of states not considered by current process.

        Args:
            param_states (dict): A mapping from param_id to its states.
        """
        for param_id, states in param_states.items():
            if param_id in self.id_to_fake_params:
                self.load_single_param_states(param_id, states)

    def optimizer_loading_epilogue(self):
        # Epilogue when loading state_dict to pytorch optimizer.
        if Version(torch.__version__) >= Version("2.0.0"):
            self.optim._patch_step_function()  # To support multiprocessing pickle/unpickle
        else:
            self.optim._hook_for_profile()  # To support multiprocessing pickle/unpickle.
        self.optim.defaults.setdefault("differentiable", False)

    def load_state_dict(self, state_dict: dict):
        """Loads optimizer state from complete optimizer state_dict.
           During loading, filter out the part of states not considered by current process.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        assert "param_groups" in state_dict
        assert "state" in state_dict
        self.load_param_groups(state_dict["param_groups"])
        self.load_param_states(state_dict["state"])
        self.optimizer_loading_epilogue()

    def state_shard(
        self, prefix: str = "", max_shard_size: int = 1024, only_rank_0: bool = True
    ) -> Iterator[Tuple[OrderedDict, int]]:
        """Returns dictionaries containing shards of optimizer states one by one.
           The max size of each dictionary shard is specified by ``max_shard_size``.

        Args:
            prefix (str, optional): the prefix for states. Default to ''.
            max_shard_size (int, optional): max size of state dict shard (in MB). Defaults to 1024.
            only_rank_0 (bool, optional): a boolean value indicating whether the state_dict is collected
                                          only on rank 0, default to True.

        Yields:
            Iterator[OrderedDict]: A generator of state dict shard of optimizer states.
        """

        sharder = StateDictSharder(max_shard_size)
        for param_id in self.id_to_real_params.keys():
            dist.barrier()
            state = self.collect_states(param_id=param_id, only_rank_0=only_rank_0)

            block, block_size = sharder.append_optim_state(param_id, state)
            if block is not None:
                yield block, block_size

        yield sharder.current_block, sharder.current_block_size

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        raise NotImplementedError("Gemini does not support clip_grad_by_value")

    def clip_grad_by_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2,
        error_if_nonfinite: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        self.logger.warning(
            f"Gemini controls grad clipping by itself, so you should not use clip_grad_by_norm", ranks=[0]
        )


class GeminiAdamOptimizer(GeminiOptimizer):
    def __init__(self, model: torch.nn.Module, **defaults: Any) -> None:
        optimizer = HybridAdam(model.parameters(), **defaults)
        super().__init__(optimizer, model, **defaults)

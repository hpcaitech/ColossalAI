from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from colossalai.utils import get_current_device
from colossalai.zero.sharded_param.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.utils.memory import colo_device_memory_capacity
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor
from colossalai.utils.memory_tracer import MemStatsCollector
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER
from typing import Type


class TensorPlacementPolicy(ABC):

    def __init__(self, device: Optional[torch.device], mem_stats_collector: Optional[MemStatsCollector] = None) -> None:
        self.device: Optional[torch.device] = device
        self.mem_stats_collector: Optional[MemStatsCollector] = mem_stats_collector

    @abstractmethod
    def evict_tensors(self, hold_cuda_tensor_list: List[StatefulTensor], **kwargs) -> None:
        raise NotImplementedError


class CPUTensorPlacementPolicy(TensorPlacementPolicy):

    def __init__(self, mem_stats_collector: Optional[MemStatsCollector] = None) -> None:
        super().__init__(torch.device('cpu'), mem_stats_collector=mem_stats_collector)

    def evict_tensors(self, hold_cuda_tensor_list: List[StatefulTensor], **kwargs) -> None:
        for t in hold_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, self.device)


class CUDATensorPlacementPolicy(TensorPlacementPolicy):

    def __init__(self, mem_stats_collector: Optional[MemStatsCollector] = None) -> None:
        assert torch.cuda.is_available(), 'Cannot use CUDATensorPlacementPolicy when CUDA is not available'
        super().__init__(get_current_device(), mem_stats_collector=mem_stats_collector)

    def evict_tensors(self, hold_cuda_tensor_list: List[StatefulTensor], **kwargs) -> None:
        pass


class AutoTensorPlacementPolicy(TensorPlacementPolicy):

    def __init__(self, mem_stats_collector: Optional[MemStatsCollector] = None) -> None:
        super().__init__(None, mem_stats_collector=mem_stats_collector)
        # model data will use 1-self._warmup_non_model_data_ratio CUDA memory in warmup phase
        self._warmup_non_model_data_ratio: float = 0.8

    def evict_tensors(self,
                      hold_cuda_tensor_list: List[StatefulTensor],
                      cuda_demand: int = 0,
                      warmup: bool = True,
                      compute_list: List[StatefulTensor] = [],
                      compute_idx: int = 0,
                      **kwargs) -> None:
        cuda_capacity = colo_device_memory_capacity(get_current_device())
        used_cuda_model_data = GLOBAL_MODEL_DATA_TRACER.cuda_usage
        if warmup:
            # We designate a part of CUDA memory for model data in warmup iterations.
            max_cuda_non_model_data_per_period = cuda_capacity * self._warmup_non_model_data_ratio
        else:
            # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
            max_cuda_non_model_data_per_period = self.mem_stats_collector.next_period_non_model_data_usage('cuda')
        total_cuda_model_data = cuda_capacity - max_cuda_non_model_data_per_period
        avail_cuda_model_data = total_cuda_model_data - used_cuda_model_data
        if avail_cuda_model_data < cuda_demand:
            # Move cuda_demand - avail_cuda_model_data volume of tensors
            # to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            freed_cuda_model_data = 0
            to_free_tensor_list = hold_cuda_tensor_list
            if not warmup:
                next_compute_idx = {t: len(compute_list) for t in hold_cuda_tensor_list}
                for i in range(len(compute_list) - 1, compute_idx, -1):
                    if compute_list[i] in next_compute_idx:
                        next_compute_idx[compute_list[i]] = i
                next_compute_idx = sorted(next_compute_idx.items(), key=lambda pair: pair[1], reverse=True)
                to_free_tensor_list = [t for (t, idx) in next_compute_idx]
            for t in to_free_tensor_list:
                if freed_cuda_model_data >= to_free_cuda_model_data:
                    break
                freed_cuda_model_data += colo_tensor_mem_usage(t)[0]
                colo_model_data_tensor_move_inline(t, torch.device('cpu'))
            if freed_cuda_model_data < to_free_cuda_model_data:
                raise RuntimeError(
                    f"Adjust layout failed! No enough CUDA memory! Need {to_free_cuda_model_data}, freed {freed_cuda_model_data}"
                )


class TensorPlacementPolicyFactory:

    @staticmethod
    def create(policy_name: str) -> Type[TensorPlacementPolicy]:
        if policy_name == 'cpu':
            return CPUTensorPlacementPolicy
        elif policy_name == 'cuda':
            return CUDATensorPlacementPolicy
        elif policy_name == 'auto':
            return AutoTensorPlacementPolicy
        else:
            raise TypeError(f"Unknown tensor placement policy {policy_name}")

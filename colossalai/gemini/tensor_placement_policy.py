from abc import ABC, abstractmethod
from time import time
from typing import List, Optional
import torch
from colossalai.utils import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity

from colossalai.gemini.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.gemini.stateful_tensor import StatefulTensor
from colossalai.gemini.memory_tracer import MemStatsCollector
from typing import Type
import functools


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

    def evict_tensors(self, hold_cuda_tensor_list: List[StatefulTensor], **kwargs) -> int:
        volume = 0
        for t in hold_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, self.device)
            volume += t.payload.numel() * t.payload.element_size()
        return volume, 0


class CUDATensorPlacementPolicy(TensorPlacementPolicy):

    def __init__(self, mem_stats_collector: Optional[MemStatsCollector] = None) -> None:
        assert torch.cuda.is_available(), 'Cannot use CUDATensorPlacementPolicy when CUDA is not available'
        super().__init__(get_current_device(), mem_stats_collector=mem_stats_collector)

    def evict_tensors(self, hold_cuda_tensor_list: List[StatefulTensor], **kwargs) -> int:
        return 0, 0


class AutoTensorPlacementPolicy(TensorPlacementPolicy):

    def __init__(self, mem_stats_collector: Optional[MemStatsCollector] = None) -> None:
        super().__init__(None, mem_stats_collector=mem_stats_collector)
        # model data will use 1-self._warmup_non_model_data_ratio CUDA memory in warmup phase
        # TODO(ver217): make these args configurable
        self._warmup_non_model_data_ratio: float = 0.8
        self._steady_cuda_cap_ratio: float = 0.9

    def evict_tensors(self,
                      hold_cuda_tensor_list: List[StatefulTensor],
                      cuda_demand: int = 0,
                      warmup: bool = True,
                      compute_list: List[StatefulTensor] = [],
                      compute_idx: int = 0,
                      **kwargs) -> int:
        """
        Evict tensors from CUDA device.

        Args:
            hold_cuda_tensor_list (List[StatefulTensor]): the list of tensor in state of HOLD-like
            cuda_demand (int, optional): the volume of data needed on cuda device. Defaults to 0.
            warmup (bool, optional): a flag indicates whether in the phase of warmup. Defaults to True.
            compute_list (List[StatefulTensor], optional): TODO. Defaults to [].
            compute_idx (int, optional): the idx of computing device. Defaults to 0.

        Raises:
            RuntimeError:

        Returns:
            int: the volume of memory that is evicted
        """
        start = time()
        cuda_capacity = colo_device_memory_capacity(get_current_device())
        used_cuda_model_data = StatefulTensor.GST_MGR.total_mem['cuda']
        if warmup:
            # We designate a part of CUDA memory for model data in warmup iterations.
            max_cuda_non_model_data_per_period = cuda_capacity * self._warmup_non_model_data_ratio
        else:
            # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
            max_cuda_non_model_data_per_period = self.mem_stats_collector.next_period_non_model_data_usage('cuda')
            cuda_capacity *= self._steady_cuda_cap_ratio
        total_cuda_model_data = cuda_capacity - max_cuda_non_model_data_per_period
        avail_cuda_model_data = total_cuda_model_data - used_cuda_model_data
        freed_cuda_model_data = 0
        end = time()
        if avail_cuda_model_data < cuda_demand:
            # Move cuda_demand - avail_cuda_model_data volume of tensors
            # to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            to_free_tensor_list = hold_cuda_tensor_list
            if not warmup:
                to_free_tensor_list = self._sort_hold_cuda_tensors(tuple(hold_cuda_tensor_list), compute_idx,
                                                                   tuple(compute_list))
                # print(self._sort_hold_cuda_tensors.cache_info())
            end = time()
            for t in to_free_tensor_list:
                if freed_cuda_model_data >= to_free_cuda_model_data:
                    break
                freed_cuda_model_data += t.payload_size
                colo_model_data_tensor_move_inline(t, torch.device('cpu'))
            if freed_cuda_model_data < to_free_cuda_model_data:
                raise RuntimeError(
                    f"Adjust layout failed! No enough CUDA memory! Need {to_free_cuda_model_data}, freed {freed_cuda_model_data}"
                )
        return freed_cuda_model_data, end - start

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _sort_hold_cuda_tensors(hold_cuda_tensors: tuple, compute_idx: int, compute_list: tuple) -> list:
        next_compute_idx = {t: len(compute_list) for t in hold_cuda_tensors}
        for i in range(len(compute_list) - 1, compute_idx, -1):
            if compute_list[i] in next_compute_idx:
                next_compute_idx[compute_list[i]] = i
        next_compute_idx = sorted(next_compute_idx.items(), key=lambda pair: pair[1], reverse=True)
        return [t for (t, idx) in next_compute_idx]


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

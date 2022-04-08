import functools
import torch
import types
from colossalai.utils.cuda import get_current_device
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor, TensorState
from colossalai.zero.shard_utils.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.utils.memory_utils.utils import colo_cuda_memory_capacity
from typing import Dict, List
from colossalai.utils.memory_tracer import MemStatsCollector
from colossalai.logging import get_dist_logger


class StatefulTensorMgr(object):
    """
    Stateful Tensor Manager, inspired from PatrickStar

    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818
    """

    def __init__(self, mem_stats_collector: MemStatsCollector) -> None:
        self._stateful_tensor_list: List[StatefulTensor] = []
        self._mem_stats_collector = mem_stats_collector
        self._logger = get_dist_logger("StatefulTensorMgr")

        self._warmup = True
        self._warmup_cuda_available_ratio = 0.2

        self._compute_list: List[StatefulTensor] = []
        self._compute_idx: int = -1

    def register_stateful_param(self, param: ShardedParamV2) -> None:
        assert isinstance(param, ShardedParamV2)
        for t in param.get_payload_tensors():
            assert isinstance(t, StatefulTensor)
            self._stateful_tensor_list.append(t)
            t.trans_state = types.MethodType(functools.partial(self._trans_state, t.trans_state), t)

    def adjust_layout(self) -> None:
        """ Adjust the layout of statefuil tensor according to the information provided
        by mem_stats_collector, which should belongs to a Sharded Model.

        Args:
            mem_stats_collector (MemStatsCollector): a collector, usually owned by a Sharded Model.
            It contains non-model footprint of a DNN model.
        """
        # find stateful tensor in state COMPUTE
        move_to_cuda_tensor_list = []
        cuda_demand = 0
        used_cuda_model_data = 0
        hold_cuda_tensor_list = []
        for tensor in self._stateful_tensor_list:
            if tensor.state == TensorState.FREE:
                continue

            if tensor.device.type == 'cuda':
                used_cuda_model_data += colo_tensor_mem_usage(tensor.payload)[0]
                if tensor.state in [TensorState.HOLD, TensorState.HOLD_AFTER_BWD, TensorState.HOLD_AFTER_FWD]:
                    hold_cuda_tensor_list.append(tensor)
            elif tensor.device.type == 'cpu':
                if tensor.state == TensorState.COMPUTE:
                    move_to_cuda_tensor_list.append(tensor)
                    cuda_demand += colo_tensor_mem_usage(tensor.payload)[1]
            else:
                raise RuntimeError
        cuda_capacity = colo_cuda_memory_capacity()

        if self._warmup:
            # We designate a part of CUDA memory for model data in warmup iterations.
            max_cuda_non_model_data_per_period = cuda_capacity * self._warmup_cuda_available_ratio
        else:
            # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
            max_cuda_non_model_data_per_period = max(self._mem_stats_collector.current_non_model_data('cuda'),
                                                     self._mem_stats_collector.next_non_model_data('cuda'))

        total_cuda_model_data = cuda_capacity - max_cuda_non_model_data_per_period
        avail_cuda_model_data = total_cuda_model_data - used_cuda_model_data

        if avail_cuda_model_data < cuda_demand:
            # Move cuda_demand - avail_cuda_model_data volume of tensors
            # to_free_cuda_model_data = cuda_demand - avail_cuda_model_data
            self.evict_tensors(hold_cuda_tensor_list, cuda_demand - avail_cuda_model_data)
        # move COMPUTE tensors to CUDA
        for t in move_to_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, get_current_device())

    def reset(self):
        """This function must be called when each iteration finishes
        """
        self._warmup = False
        self._compute_idx = -1

    def evict_tensors(self, hold_cuda_tensor_list, to_free_cuda_model_data):
        freed_cuda_model_data = 0
        to_free_tensor_list = hold_cuda_tensor_list
        if not self._warmup:
            next_compute_idx: Dict[StatefulTensor, int] = {t: len(self._compute_list) for t in hold_cuda_tensor_list}
            for i in range(len(self._compute_list) - 1, self._compute_idx, -1):
                if self._compute_list[i] in next_compute_idx:
                    next_compute_idx[self._compute_list[i]] = i
            next_compute_idx = sorted(next_compute_idx.items(), key=lambda pair: pair[1], reverse=True)
            to_free_tensor_list = [t for (t, idx) in next_compute_idx]
        for t in to_free_tensor_list:
            if freed_cuda_model_data > to_free_cuda_model_data:
                break
            freed_cuda_model_data += colo_tensor_mem_usage(t)[0]
            colo_model_data_tensor_move_inline(t, torch.device('cpu'))
        if freed_cuda_model_data < to_free_cuda_model_data:
            raise RuntimeError(
                f"Adjust layout failed! No enough CUDA memory! Need {to_free_cuda_model_data}, freed {freed_cuda_model_data}"
            )

    def _trans_state(self, trans_state_func, stateful_tensor, state):
        trans_state_func(state)
        if state == TensorState.COMPUTE:
            self._compute_idx += 1
            if self._warmup:
                self._compute_list.append(stateful_tensor)

import functools
import types
from time import time
from typing import List

from colossalai.accelerator import get_accelerator

from .stateful_tensor import StatefulTensor, TensorState
from .tensor_placement_policy import TensorPlacementPolicy
from .tensor_utils import colo_model_data_tensor_move_inline


class StatefulTensorMgr(object):
    """
    Stateful Tensor Manager, inspired from PatrickStar

    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818
    """

    def __init__(self, tensor_placement_policy: TensorPlacementPolicy) -> None:
        self._tensor_placement_policy: TensorPlacementPolicy = tensor_placement_policy
        self._stateful_tensor_list: List[StatefulTensor] = []

        self._compute_list: List[StatefulTensor] = []
        self._compute_idx: int = -1

        self._cpu_gpu_move_volume = 0
        self._layout_time = 0
        self._evict_time = 0
        self._warmup = True

    def register_stateful_tensor_list(self, tensor_list: List[StatefulTensor]) -> None:
        assert self._stateful_tensor_list == [], "Can't register stateful tensors for manager twice"
        self._stateful_tensor_list = tensor_list
        for t in self._stateful_tensor_list:
            assert isinstance(t, StatefulTensor)
            t.trans_state = types.MethodType(functools.partial(self._trans_state, t.trans_state), t)

    def start_iter(self):
        pass

    def finish_iter(self):
        """This function must be called when each iteration finishes"""
        self._warmup = False
        self._compute_idx = -1
        self._cpu_gpu_move_volume = 0
        self._layout_time = 0
        self._evict_time = 0

    def adjust_layout(self) -> None:
        """Adjust the layout of stateful tensor according to the information provided
        by mem_stats_collector, which should belongs to a Sharded Model.
        """
        # find stateful tensor in state COMPUTE
        cuda_demand = StatefulTensor.GST_MGR.state_mem["cpu"][TensorState.COMPUTE]
        start = time()
        move_to_cuda_tensor_list, hold_cuda_tensor_list = self._get_layout_info(self._compute_idx, self._warmup)
        self._layout_time += time() - start
        vol, evict_time = self._tensor_placement_policy.evict_tensors(
            hold_cuda_tensor_list,
            cuda_demand=cuda_demand,
            warmup=self._warmup,
            compute_list=self._compute_list,
            compute_idx=self._compute_idx,
        )
        self._cpu_gpu_move_volume += vol
        self._evict_time += evict_time
        # move COMPUTE tensors to CUDA
        self._cpu_gpu_move_volume += cuda_demand
        for t in move_to_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, get_accelerator().get_current_device())

    @property
    def cpu_gpu_move_volume(self):
        return self._cpu_gpu_move_volume

    def _trans_state(self, trans_state_func, stateful_tensor, state):
        trans_state_func(state)
        if state == TensorState.COMPUTE:
            self._compute_idx += 1
            if self._warmup:
                self._compute_list.append(stateful_tensor)

    @functools.lru_cache(maxsize=None)
    def _get_layout_info(self, compute_idx: int, warmup: bool):
        move_to_cuda_tensor_list = []
        hold_cuda_tensor_list = []
        for tensor in self._stateful_tensor_list:
            if tensor.state == TensorState.FREE:
                continue

            if tensor.device.type == "cuda":
                if tensor.state in [TensorState.HOLD, TensorState.HOLD_AFTER_BWD, TensorState.HOLD_AFTER_FWD]:
                    hold_cuda_tensor_list.append(tensor)
            elif tensor.device.type == "cpu":
                if tensor.state == TensorState.COMPUTE:
                    move_to_cuda_tensor_list.append(tensor)
            else:
                raise RuntimeError
        return move_to_cuda_tensor_list, hold_cuda_tensor_list

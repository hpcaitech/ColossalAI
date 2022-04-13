import functools
import torch
import types
from colossalai.utils.cuda import get_current_device
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor, TensorState
from colossalai.zero.sharded_param.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.zero.utils.tensor_placement_policy import TensorPlacementPolicy
from typing import List
from colossalai.logging import get_dist_logger


class StatefulTensorMgr(object):
    """
    Stateful Tensor Manager, inspired from PatrickStar

    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818
    """

    def __init__(self, tensor_placement_policy: TensorPlacementPolicy) -> None:
        self._tensor_placement_policy: TensorPlacementPolicy = tensor_placement_policy
        self._stateful_tensor_list: List[StatefulTensor] = []
        self._logger = get_dist_logger("StatefulTensorMgr")

        self._warmup = True

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
        cuda_demand = 0
        move_to_cuda_tensor_list = []
        hold_cuda_tensor_list = []
        for tensor in self._stateful_tensor_list:
            if tensor.state == TensorState.FREE:
                continue

            if tensor.device.type == 'cuda':
                if tensor.state in [TensorState.HOLD, TensorState.HOLD_AFTER_BWD, TensorState.HOLD_AFTER_FWD]:
                    hold_cuda_tensor_list.append(tensor)
            elif tensor.device.type == 'cpu':
                if tensor.state == TensorState.COMPUTE:
                    move_to_cuda_tensor_list.append(tensor)
                    cuda_demand += colo_tensor_mem_usage(tensor.payload)[1]
            else:
                raise RuntimeError
        self._tensor_placement_policy.evict_tensors(hold_cuda_tensor_list,
                                                    cuda_demand=cuda_demand,
                                                    warmup=self._warmup,
                                                    compute_list=self._compute_list,
                                                    compute_idx=self._compute_idx)
        # move COMPUTE tensors to CUDA
        for t in move_to_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, get_current_device())

    def reset(self):
        """This function must be called when each iteration finishes
        """
        self._warmup = False
        self._compute_idx = -1

    def _trans_state(self, trans_state_func, stateful_tensor, state):
        trans_state_func(state)
        if state == TensorState.COMPUTE:
            self._compute_idx += 1
            if self._warmup:
                self._compute_list.append(stateful_tensor)

import torch
from colossalai.utils.cuda import get_current_device
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor, TensorState
from colossalai.zero.shard_utils.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.utils.memory_utils.utils import colo_cuda_memory_capacity
from typing import Set
from colossalai.utils.memory_tracer import MemStatsCollector
from colossalai.logging import get_dist_logger


class StatefulTensorMgr(object):
    """
    Stateful Tensor Manager, inspired from PatrickStar
    
    PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management
    https://arxiv.org/abs/2108.05818
    """

    def __init__(self, mem_stats_collector: MemStatsCollector) -> None:
        self._stateful_tensor_list: Set[ShardedParamV2] = set()
        self._mem_stats_collector = mem_stats_collector
        self._logger = get_dist_logger("StatefulTensorMgr")

        self._warmup = True
        self._warmup_cuda_available_ratio = 0.2

    def register_stateful_param(self, param: ShardedParamV2) -> None:
        assert isinstance(param, ShardedParamV2)
        for t in param.get_payload_tensors():
            assert isinstance(t, StatefulTensor)
            self._stateful_tensor_list.add(t)

    def adjust_layout(self) -> None:
        """ Adjust the layout of statefuil tensor according to the information provided
        by mem_stats_collector, which should belongs to a Sharded Model.

        Args:
            mem_stats_collector (MemStatsCollector): a collector, usually owned by a Sharded Model.
            It contains non-model footprint of a DNN model.
        """
        # find stateful tensor in state COMPUTE
        # self._logger.info("Adjust Tensor Layout Begin", ranks=[0])

        move_to_cuda_tensor_list = []
        cuda_demand = 0
        used_cuda_model_data = 0
        hold_cuda_tensor_list = []

        self._logger.info(f"stateful tensor num {len(self._stateful_tensor_list)}", ranks=[0])
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
                    cuda_demand += colo_tensor_mem_usage(tensor.payload)[0]
            else:
                raise RuntimeError
        cuda_capacity = colo_cuda_memory_capacity()

        self._logger.info(f"move_to_cuda_tensor_list len {len(move_to_cuda_tensor_list)}")
        if self._warmup:
            # We designate a part of CUDA memory for model data in warmup iterations.
            max_cuda_non_model_data_per_period = cuda_capacity * self._warmup_cuda_available_ratio
        else:
            # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
            max_cuda_non_model_data_per_period = max(self._mem_stats_collector.current_non_model_data('cuda'),
                                                     self._mem_stats_collector.next_non_model_data('cuda'))

        cuda_model_data_period = cuda_capacity - max_cuda_non_model_data_per_period

        if cuda_model_data_period < used_cuda_model_data + cuda_demand:
            # move cuda_model_data_period - cuda_demand - used_cuda_model_data volume of tensor
            # Here use a naive eviction strategy.
            acc_size = 0
            for t in hold_cuda_tensor_list:
                if acc_size > cuda_demand:
                    break
                colo_model_data_tensor_move_inline(t, torch.device('cpu'))
                self._logger.info(f"move tensor cuda -> cpu", ranks=[0])
                t_size = colo_tensor_mem_usage(t)
                acc_size += t_size
            if acc_size < cuda_demand:
                raise RuntimeError("Adjust layout failed! No enough CUDA memory!")

        # move COMPUTE tensors to CUDA
        for t in move_to_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, get_current_device())
            self._logger.info(f"move tensor cpu -> cuda", ranks=[0])
        self._logger.info("Adjust Tensor Layout Finished", ranks=[0])

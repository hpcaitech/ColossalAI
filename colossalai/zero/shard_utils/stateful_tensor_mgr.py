import torch
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.utils.cuda import get_current_device
from colossalai.zero.sharded_param.sharded_param import ShardedParamV2
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor, TensorState
from colossalai.zero.shard_utils.tensor_utils import colo_model_data_tensor_move_inline, colo_tensor_mem_usage
from colossalai.utils.memory_utils.utils import colo_cuda_memory_capacity
from typing import Set
from colossalai.utils.memory_tracer import MemStatsCollector


class StatefulTensorMgr(SingletonMeta):
    _stateful_tensor_list: Set[ShardedParamV2] = set()

    def register_param(self, param: ShardedParamV2) -> None:
        for t in param.get_payload_tensors():
            assert isinstance(t, StatefulTensor)
            self._stateful_tensor_list.add(t)

    def evict_tensors(self) -> None:
        pass

    def adjust_layout(self, mem_stats_collector: MemStatsCollector) -> None:
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
            else:
                if tensor.state == TensorState.COMPUTE:
                    move_to_cuda_tensor_list.append(tensor)
                    cuda_demand += colo_tensor_mem_usage(tensor.payload)[0]

        # max non-model-data cuda memory consumption of this sampling moment and the next sampling moment.
        max_cuda_non_model_data_per_period = max(mem_stats_collector.current_non_model_data('cuda'),
                                                 mem_stats_collector.next_non_model_data('cuda'))
        cuda_capacity = colo_cuda_memory_capacity()
        cuda_model_data_period = cuda_capacity - max_cuda_non_model_data_per_period
        if cuda_model_data_period < used_cuda_model_data + cuda_demand:
            # move cuda_model_data_period - cuda_demand - used_cuda_model_data volume of tensor
            # Here use a naive eviction strategy.
            acc_size = 0
            for t in hold_cuda_tensor_list:
                if acc_size > cuda_demand:
                    break
                colo_model_data_tensor_move_inline(t, torch.device('cpu'))
                t_size = colo_tensor_mem_usage(t)
                acc_size += t_size
            if acc_size < cuda_demand:
                raise RuntimeError("Adjust layout failed! No enough CUDA memory!")

        # move COMPUTE tensors to CUDA
        for t in move_to_cuda_tensor_list:
            colo_model_data_tensor_move_inline(t, get_current_device())

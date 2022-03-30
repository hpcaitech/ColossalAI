from typing import Optional

import torch
import torch.distributed as dist
from colossalai.registry import OPHOOKS
from colossalai.utils import get_current_device
from colossalai.utils.memory_tracer.memstats_collector import MemStatsCollector
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_param.tensorful_state import TensorState

from ._base_ophook import BaseOpHook

from colossalai.utils.memory_utils.utils import \
    colo_model_data_tensor_move_inline


@OPHOOKS.register_module
class ZeroHook(BaseOpHook):
    """
    A hook to process sharded param for ZeRO method.
    """

    def __init__(self,
                 shard_strategy: BaseShardStrategy,
                 memstarts_collector: Optional[MemStatsCollector],
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.shard_strategy = shard_strategy
        self.process_group = process_group
        # NOTE(jiaruifang) Now the computing device of FWD and BWD is always on GPU
        self.computing_device = torch.device(f'cuda:{get_current_device()}')

        self._memstarts_collector = memstarts_collector

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        tensor_list = []
        for param in module.parameters(recurse=False):
            assert hasattr(param, 'col_attr')
            tensor_list.append(param.col_attr.sharded_data_tensor)
        self.shard_strategy.gather(tensor_list, self.process_group)
        for param in module.parameters(recurse=False):
            colo_model_data_tensor_move_inline(param.col_attr.sharded_data_tensor, self.computing_device)
            param.data = param.col_attr.sharded_data_tensor.payload

        if self._memstarts_collector:
            self._memstarts_collector.sample_memstats()

        for param in module.parameters(recurse=False):
            param.col_attr.sharded_data_tensor.trans_state(TensorState.COMPUTE)

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        for param in module.parameters(recurse=False):
            param.col_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_FWD)

        tensor_list = []
        for param in module.parameters(recurse=False):
            assert hasattr(param, 'col_attr')
            tensor_list.append(param.col_attr.sharded_data_tensor)
        self.shard_strategy.shard(tensor_list, self.process_group)
        for param in module.parameters(recurse=False):
            param.col_attr.remove_torch_payload()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        tensor_list = []
        for param in module.parameters(recurse=False):
            assert hasattr(param, 'col_attr')
            tensor_list.append(param.col_attr.sharded_data_tensor)
        self.shard_strategy.gather(tensor_list, self.process_group)
        for param in module.parameters(recurse=False):
            colo_model_data_tensor_move_inline(param.col_attr.sharded_data_tensor, self.computing_device)
            param.data = param.col_attr.sharded_data_tensor.payload
        if self._memstarts_collector:
            self._memstarts_collector.sample_memstats()

        for param in module.parameters(recurse=False):
            param.col_attr.sharded_data_tensor.trans_state(TensorState.COMPUTE)

    def post_bwd_exec(self, module: torch.nn.Module, input):
        for param in module.parameters(recurse=False):
            param.col_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_BWD)

        tensor_list = []
        for param in module.parameters(recurse=False):
            assert hasattr(param, 'col_attr')
            tensor_list.append(param.col_attr.sharded_data_tensor)
        self.shard_strategy.shard(tensor_list, self.process_group)

        for param in module.parameters(recurse=False):
            param.col_attr.remove_torch_payload()

    def pre_iter(self):
        pass

    def post_iter(self):
        pass

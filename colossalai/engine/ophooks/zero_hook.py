from typing import Optional

import torch
import torch.distributed as dist
from colossalai.registry import OPHOOKS
from colossalai.utils import get_current_device
from colossalai.utils.memory_tracer.memstats_collector import MemStatsCollector
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_param.tensorful_state import TensorState
from colossalai.zero.shard_utils.stateful_tensor_mgr import StatefulTensorMgr

from ._base_ophook import BaseOpHook

from colossalai.zero.shard_utils.tensor_utils import colo_model_data_tensor_move_inline


@OPHOOKS.register_module
class ZeroHook(BaseOpHook):
    """
    A hook to process sharded param for ZeRO method.
    """

    def __init__(self,
                 shard_strategy: BaseShardStrategy,
                 memstarts_collector: Optional[MemStatsCollector] = None,
                 stateful_tensor_mgr: Optional[StatefulTensorMgr] = None,
                 process_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.shard_strategy = shard_strategy
        self.process_group = process_group

        # NOTE(jiaruifang) Now the computing device of FWD and BWD is always on GPU
        self.computing_device = get_current_device()

        self._memstarts_collector = memstarts_collector
        self._stateful_tensor_mgr = stateful_tensor_mgr

    def pre_fwd_exec(self, module: torch.nn.Module, *args):

        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.COMPUTE)

        if self._stateful_tensor_mgr:
            self._stateful_tensor_mgr.adjust_layout()
        else:
            for param in module.parameters(recurse=False):
                colo_model_data_tensor_move_inline(param.colo_attr.sharded_data_tensor, self.computing_device)

        # gather sharded parameters
        if module.param_is_sharded:
            tensor_list = []
            for param in module.parameters(recurse=False):
                assert hasattr(param, 'colo_attr')
                tensor_list.append(param.colo_attr.sharded_data_tensor)
            self.shard_strategy.gather(tensor_list, self.process_group)

        # record memory statistics
        if self._memstarts_collector:
            self._memstarts_collector.sample_memstats()

        for param in module.parameters(recurse=False):
            param.data = param.colo_attr.sharded_data_tensor.payload
            assert param.data.device.type == 'cuda', f"PRE FWD param.data must be on CUDA"

    def post_fwd_exec(self, module: torch.nn.Module, *args):

        # change tensor state to HOLD_AFTER_FWD
        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_FWD)

        # shard gathered parameters
        if module.param_is_sharded:
            tensor_list = []
            for param in module.parameters(recurse=False):
                assert hasattr(param, 'colo_attr')
                tensor_list.append(param.colo_attr.sharded_data_tensor)
            self.shard_strategy.shard(tensor_list, self.process_group)

        # remove torch payload
        for param in module.parameters(recurse=False):
            param.colo_attr.remove_torch_payload()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):

        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.COMPUTE)

        if self._stateful_tensor_mgr:
            self._stateful_tensor_mgr.adjust_layout()
        else:
            for param in module.parameters(recurse=False):
                colo_model_data_tensor_move_inline(param.colo_attr.sharded_data_tensor, self.computing_device)

        # gather sharded parameters
        if module.param_is_sharded:
            tensor_list = []
            for param in module.parameters(recurse=False):
                assert hasattr(param, 'colo_attr')
                tensor_list.append(param.colo_attr.sharded_data_tensor)
            self.shard_strategy.gather(tensor_list, self.process_group)

        # record memory statistics
        if self._memstarts_collector:
            self._memstarts_collector.sample_memstats()

        for param in module.parameters(recurse=False):
            param.data = param.colo_attr.sharded_data_tensor.payload
            assert param.data.device.type == 'cuda', f"PRE BWD param.data must be on CUDA"

    def post_bwd_exec(self, module: torch.nn.Module, input):

        # change tensor state to HOLD_AFTER_BWD
        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_BWD)

        # shard gathered parameters
        if module.param_is_sharded:
            tensor_list = []
            for param in module.parameters(recurse=False):
                assert hasattr(param, 'colo_attr')
                tensor_list.append(param.colo_attr.sharded_data_tensor)
            self.shard_strategy.shard(tensor_list, self.process_group)

        # remove torch payload
        for param in module.parameters(recurse=False):
            param.colo_attr.remove_torch_payload()

    def pre_iter(self):
        pass

    def post_iter(self):
        if self._stateful_tensor_mgr:
            self._stateful_tensor_mgr.reset()

from typing import Optional

import torch
import torch.distributed as dist

from colossalai.accelerator import get_accelerator
from colossalai.legacy.registry import OPHOOKS
from colossalai.legacy.zero.gemini.ophooks import BaseOpHook
from colossalai.legacy.zero.gemini.stateful_tensor import TensorState
from colossalai.legacy.zero.gemini.stateful_tensor_mgr import StatefulTensorMgr
from colossalai.legacy.zero.shard_utils import BaseShardStrategy
from colossalai.logging import get_dist_logger
from colossalai.zero.gemini.memory_tracer import MemStatsCollector


@OPHOOKS.register_module
class ZeroHook(BaseOpHook):
    """
    A hook to process sharded param for ZeRO method.
    Warning: this class has been deprecated after version 0.1.12
    """

    def __init__(
        self,
        shard_strategy: BaseShardStrategy,
        memstarts_collector: Optional[MemStatsCollector] = None,
        stateful_tensor_mgr: Optional[StatefulTensorMgr] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.logger = get_dist_logger("ZeROHook")
        self.shard_strategy = shard_strategy
        self.process_group = process_group

        # NOTE(jiaruifang) Now the computing device of FWD and BWD is always on GPU
        self.computing_device = get_accelerator().get_current_device()

        self._memstarts_collector = memstarts_collector
        self._stateful_tensor_mgr = stateful_tensor_mgr

    def gather_parameters(self, module: torch.nn.Module):
        # gather sharded parameters
        if module.param_is_sharded:
            tensor_list = []
            for param in module.parameters(recurse=False):
                assert hasattr(param, "colo_attr")
                tensor_list.append(param.colo_attr.sharded_data_tensor)
            self.shard_strategy.gather(tensor_list, self.process_group)

    def shard_parameters(self, module: torch.nn.Module):
        # shard gathered parameters
        if module.param_is_sharded:
            tensor_list = []
            for param in module.parameters(recurse=False):
                assert hasattr(param, "colo_attr")
                tensor_list.append(param.colo_attr.sharded_data_tensor)
            self.shard_strategy.shard(tensor_list, self.process_group)

    def adjust_module_data(self, module: torch.nn.Module):
        # record overall data statistics
        if self._memstarts_collector:
            self._memstarts_collector.sample_overall_data()

        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.COMPUTE)

        # adjust stateful tensor to get enough CUDA memory
        self._stateful_tensor_mgr.adjust_layout()

        # record model data statistics
        if self._memstarts_collector:
            self._memstarts_collector.record_model_data_volume()

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        self.adjust_module_data(module)
        self.gather_parameters(module)
        for param in module.parameters(recurse=False):
            param.data = param.colo_attr.data_payload
            assert param.data.device.type == "cuda", f"PRE FWD param.data must be on CUDA"

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        # change tensor state to HOLD_AFTER_FWD
        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_FWD)

        self.shard_parameters(module)

        # remove torch payload
        for param in module.parameters(recurse=False):
            param.colo_attr.set_data_none()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        self.adjust_module_data(module)
        self.gather_parameters(module)
        for param in module.parameters(recurse=False):
            param.data = param.colo_attr.data_payload
            assert param.data.device.type == "cuda", f"PRE BWD param.data must be on CUDA"

    def post_bwd_exec(self, module: torch.nn.Module, input):
        # change tensor state to HOLD_AFTER_BWD
        for param in module.parameters(recurse=False):
            param.colo_attr.sharded_data_tensor.trans_state(TensorState.HOLD_AFTER_BWD)

        self.shard_parameters(module)

        # remove torch payload
        for param in module.parameters(recurse=False):
            param.colo_attr.set_data_none()

    def pre_iter(self):
        pass

    def post_iter(self):
        if self._stateful_tensor_mgr:
            self.logger.debug(
                f"CPU-GPU data moving this iteration {self._stateful_tensor_mgr.cpu_gpu_move_volume/1e9} GB, get layout info time: {self._stateful_tensor_mgr._layout_time}, evict cpu time: {self._stateful_tensor_mgr._evict_time}",
                ranks=[0],
            )
            self._stateful_tensor_mgr.finish_iter()

from typing import Optional

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp, is_compatible_with_meta
from colossalai.zero.gemini.chunk import ChunkManager

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

from .chunk_memstats_collector import ChunkMemStatsCollector


class ModuleInfos:
    def __init__(
        self, module: torch.nn.Module, module_name: str, module_full_name: str, parent_module: torch.nn.Module
    ):
        self.module = module
        self.module_name = module_name
        self.module_full_name = module_full_name
        self.parent_module = parent_module


class StaticMemStatsCollector(ChunkMemStatsCollector):
    """
    A Static Memory statistic collector.
    """

    def __init__(self, module: nn.Module, chunk_manager: ChunkManager) -> None:
        super().__init__(chunk_manager)
        self.module = module
        self.module_info_list = []

    def init_mem_stats(self, *inputs):
        self.register_opnodes_recursively(self.module)
        self.refactor_module()

        self.module = self.module.cpu()
        self.module.train()

        data = [MetaTensor(torch.rand(inp.shape, device="meta"), fake_device="cpu") for inp in inputs]
        gm = symbolic_trace(self.module)
        interp = MetaInfoProp(gm)
        interp.propagate(*data)

        total_mem = 0
        for inp in inputs:
            total_mem += inp.numel() * inp.element_size()
        last_node = None
        module_name_list = [mInfo.module_full_name for mInfo in self.module_info_list]
        for node in gm.graph.nodes:
            total_mem = total_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            if node.op == "call_module":
                if node.name.endswith("_0") and node.name[:-2] in module_name_list:
                    self._non_model_data_cuda_list.append(total_mem)
                last_node = node
        self._non_model_data_cuda_list.append(total_mem)
        self._non_model_data_cuda_list = self._non_model_data_cuda_list[1:]

        cur_module_mem_fwd = 0
        cur_module_mem_bwd = 0
        grad_module_out = last_node.meta["fwd_mem_out"]
        for node in gm.graph.nodes.__reversed__():
            cur_module_mem_fwd = cur_module_mem_fwd + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            cur_module_mem_bwd = cur_module_mem_bwd + node.meta["bwd_mem_tmp"] + node.meta["bwd_mem_out"]
            if node.op == "call_module":
                if node.name.endswith("_0") and node.name[:-2] in module_name_list:
                    self._non_model_data_cuda_list.append(total_mem + grad_module_out + cur_module_mem_bwd)
                    total_mem = total_mem - cur_module_mem_fwd
                    cur_module_mem_fwd = 0
                    cur_module_mem_bwd = 0
                    grad_module_out = node.meta["bwd_mem_out"]

        self._step_total = len(self._non_model_data_cuda_list)
        self.recover_module()

    def refactor_module(self):
        for modInfo in self.module_info_list:
            temp_node = nn.Sequential(nn.ReLU(), modInfo.module)
            modInfo.parent_module.__setattr__(modInfo.module_name, temp_node)

    def recover_module(self):
        for modInfo in self.module_info_list:
            modInfo.parent_module.__setattr__(modInfo.module_name, modInfo.module)

    def register_opnodes_recursively(
        self,
        module: torch.nn.Module,
        name: str = "",
        full_name: str = "",
        parent_module: Optional[torch.nn.Module] = None,
    ):
        assert isinstance(module, torch.nn.Module)

        for child_name, child in module.named_children():
            self.register_opnodes_recursively(child, child_name, full_name + "_" + child_name, module)

        # Early return on modules with no parameters.
        if len(list(module.parameters(recurse=False))) == 0:
            return

        self.module_info_list.append(ModuleInfos(module, name, full_name[1:], parent_module))

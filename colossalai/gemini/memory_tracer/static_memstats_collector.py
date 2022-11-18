from typing import Optional

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp, is_compatible_with_meta

if is_compatible_with_meta():
    from colossalai.fx.profiler import MetaTensor

from .memstats_collector import MemStatsCollector


class ModuleInfos:

    def __init__(self, module: torch.nn.Module, module_name: str, module_full_name: str,
                 parent_module: torch.nn.Module):
        self.module = module
        self.module_name = module_name
        self.module_full_name = module_full_name
        self.parent_module = parent_module


class StaticMemStatsCollector(MemStatsCollector):
    """
    A Static Memory statistic collector.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.module_info_list = []

    def init_mem_stats(self, **kwargs):

        self.register_opnodes_recursively(self.module)
        self.refactor_module()

        self.module = self.module.cpu()
        self.module.train()

        graph = ColoTracer().trace(self.module, meta_args=kwargs)
        gm = torch.fx.GraphModule(self.module, graph)
        interp = MetaInfoProp(gm)
        interp.propagate(*[MetaTensor(v, fake_device='cpu') for k, v in kwargs.items()])

        module_name_list = [mInfo.module_full_name for mInfo in self.module_info_list]
        fwd_out_released = {}
        total_mem = 0

        # forward
        for node in gm.graph.nodes:
            total_mem = total_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            if calculate_fwd_out(node) > 0:
                fwd_out_released[node] = False
            if node.op == "call_module" and isinstance(node.target, str):
                module_name = node.target.replace(".", "_")
                if module_name.endswith("_0") and module_name[:-2] in module_name_list:
                    self._non_model_data_cuda_list.append(total_mem)
                    node.meta["bwd_mem_tmp"] = 0
                    node.meta["bwd_mem_out"] = 0

        self._non_model_data_cuda_list.append(total_mem)
        self._non_model_data_cuda_list = self._non_model_data_cuda_list[1:]

        peak_mem = total_mem
        grad_in_computed = {}

        # backward
        for node in gm.graph.nodes.__reversed__():

            if node.name.__contains__("where") or node.name.__contains__("truediv"):
                continue

            # before run backward of the node

            total_mem = total_mem + node.meta["bwd_mem_tmp"] + node.meta["bwd_mem_out"]
            if total_mem >= peak_mem:
                peak_mem = total_mem

            # after run backward of the node

            # release temp memory
            total_mem -= node.meta["bwd_mem_tmp"]
            total_mem -= calculate_fwd_tmp(node)

            # release grad_in of current node
            for grad_in in node.meta["fwd_out"]:
                if isinstance(grad_in, torch.Tensor):
                    total_mem -= grad_in.numel() * torch.tensor([], dtype=grad_in.dtype).element_size()

            for in_node in node.args:
                if isinstance(in_node, torch.fx.node.Node):
                    # release fwd_in (fwd_out) of current node (input nodes)
                    if calculate_fwd_out(in_node) > 0 and (not fwd_out_released[in_node]):
                        total_mem -= calculate_fwd_out(in_node)
                        fwd_out_released[in_node] = True
                    # map multiple gradients of output to one tensor
                    if grad_in_computed.get(in_node, False):
                        total_mem -= calculate_fwd_out(in_node)
                        grad_in_computed[in_node] = True

            if node.name == "output":
                for in_node in node.args:
                    if isinstance(in_node, torch.fx.node.Node):
                        total_mem += calculate_fwd_out(in_node)

            if node.op == "call_module" and isinstance(node.target, str):
                module_name = node.target.replace(".", "_")
                if module_name.endswith("_0") and module_name[:-2] in module_name_list:
                    self._non_model_data_cuda_list.append(peak_mem)
                    # add grad_in of Identity module
                    for grad_in in node.meta["fwd_out"]:
                        if isinstance(grad_in, torch.Tensor):
                            total_mem += grad_in.numel() * torch.tensor([], dtype=grad_in.dtype).element_size()
                    peak_mem = total_mem

        self._step_total = len(self._non_model_data_cuda_list)
        self.recover_module()

    def refactor_module(self):
        for modInfo in self.module_info_list:
            temp_node = nn.Sequential(nn.Identity(), modInfo.module)
            modInfo.parent_module.__setattr__(modInfo.module_name, temp_node)

    def recover_module(self):
        for modInfo in self.module_info_list:
            modInfo.parent_module.__setattr__(modInfo.module_name, modInfo.module)

    def register_opnodes_recursively(self,
                                     module: torch.nn.Module,
                                     name: str = "",
                                     full_name: str = "",
                                     parent_module: Optional[torch.nn.Module] = None):

        assert isinstance(module, torch.nn.Module)

        for child_name, child in module.named_children():
            self.register_opnodes_recursively(child, child_name, full_name + "_" + child_name, module)

        # Early return on modules with no parameters.
        if len(list(module.parameters(recurse=False))) == 0:
            return

        self.module_info_list.append(ModuleInfos(module, name, full_name[1:], parent_module))

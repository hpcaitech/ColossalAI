from dataclasses import dataclass
from typing import List

import torch

from colossalai.context.singleton_meta import SingletonMeta
from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp

from .region import Region


@dataclass
class NodeInfo:
    node_id: int = 0
    runtime_fwd_mem: float = 0
    runtime_bwd_mem: float = 0


class NvDevicePower:
    """
    NVIDIA GPU computing performance (TFLOPs).
    """

    RTX3080_FP16 = 70
    RTX3080_FP32 = 34.1

    RTX3090_FP16 = 71
    RTX3090_FP32 = 35.7

    V100_FP16 = 31.4
    V100_FP32 = 15.7

    A100_FP16 = 78
    A100_FP32 = 19.5


class GlobalRuntimeInfo(metaclass=SingletonMeta):
    def __init__(self):
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()
        self.fwd_prefetch_event_map = {}
        self.bwd_prefetch_event_map = {}
        self.region_list = []


def compute_act_peak_mem(region_list: List[Region]) -> float:
    act_peak_mem = 0
    runtime_mem = 0
    # forward
    for region in region_list:
        for node in region.nodes:
            runtime_mem = runtime_mem + calculate_fwd_tmp(node) + calculate_fwd_out(node)
            act_peak_mem = max(runtime_mem, act_peak_mem)
    # backward
    bwd_deps = {}
    for region in region_list.__reversed__():
        for node in region.nodes.__reversed__():
            runtime_mem -= calculate_fwd_out(node)
            runtime_mem = runtime_mem + node.meta["bwd_mem_tmp"] + node.meta["bwd_mem_out"]

            act_peak_mem = max(runtime_mem, act_peak_mem)

            runtime_mem = runtime_mem - node.meta["bwd_mem_tmp"] - calculate_fwd_tmp(node)

            # free bwd_mem_out
            bwd_deps[node] = len(node.all_input_nodes)
            for user_node in node.users:
                if user_node in bwd_deps:
                    bwd_deps[user_node] -= 1
                    if bwd_deps[user_node] <= 0:
                        runtime_mem -= user_node.meta["bwd_mem_out"]

    return act_peak_mem


def compute_max_param_mem(region_list: List[Region]) -> float:
    return max(region.param_size for region in region_list)


def compute_total_param_mem(region_list: List[Region]) -> float:
    return sum(region.param_size for region in region_list if region.r_id <= region.shared_rid)


def requires_upload_p_in_fwd(shared_reg: Region):
    return (shared_reg.r_id >= shared_reg.shared_rid) or (
        shared_reg.r_id < shared_reg.shared_rid and shared_reg.need_offload
    )


def requires_release_p_in_bwd(shared_reg: Region):
    return (shared_reg.r_id >= shared_reg.shared_rid) or (
        shared_reg.r_id < shared_reg.shared_rid and shared_reg.need_offload
    )


def requires_offload_g_in_bwd(region: Region):
    return region.param_size and (region.r_id <= region.shared_rid)

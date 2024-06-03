from typing import Dict

import torch
import torch.fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_map

from colossalai.fx import ColoTracer, is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp

from .base_offload_module import BaseOffloadModule
from .region_manager import RegionManager
from .runtime import runtime_asyn_offload_apply_pass, runtime_syn_offload_apply_pass
from .util import GlobalRuntimeInfo, compute_act_peak_mem, compute_max_param_mem, compute_total_param_mem


def memory_optimize(
    model: torch.nn.Module, inps: Dict[str, torch.Tensor], memory_budget: float = -1.0, solver_name: str = "asyn"
):
    model = model.cpu().half()
    tracer = ColoTracer()
    assert is_compatible_with_meta()
    wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, inps)
    graph = tracer.trace(model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)
    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    region_manager = RegionManager(graph, solver_name=solver_name, memory_budget=memory_budget)
    region_manager._build_regions()
    GlobalRuntimeInfo().region_list = region_manager.region_list

    act_peak_mem = compute_act_peak_mem(region_manager.region_list) / 1024**2
    max_param_mem = compute_max_param_mem(region_manager.region_list) / 1024**2
    total_param_mem = compute_total_param_mem(region_manager.region_list) / 1024**2
    print(
        f"act_peak_mem={act_peak_mem:.3f} MB | max_param_mem={max_param_mem:.3f} MB | total_param_mem={total_param_mem:.3f}"
    )

    if solver_name == "syn":
        gm = runtime_syn_offload_apply_pass(gm, region_manager.region_list)
    elif solver_name == "asyn":
        gm = runtime_asyn_offload_apply_pass(gm, region_manager.region_list)
    else:
        raise TypeError(f"Unknown solver name {solver_name}!")

    gm.recompile()
    optimized_model = BaseOffloadModule(gm, region_manager, solver_name == "syn")
    return optimized_model

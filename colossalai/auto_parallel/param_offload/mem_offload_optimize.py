from typing import List, Dict
import torch
import torch.fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_map

from colossalai.fx import ColoTracer, is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp

from colossalai.auto_parallel.param_offload.strategies_constructor import OffloadStrategiesConstructor
from colossalai.auto_parallel.param_offload.solver import Solver
from colossalai.auto_parallel.param_offload.runtime import runtime_offload_apply_pass
from colossalai.auto_parallel.param_offload.basic_offload_module import OffloadModuleWrapper


def memory_optimization(model: torch.nn.Module, inps: Dict[str, torch.Tensor], memory_budget: float=-1.0):
    model.cpu()
    tracer = ColoTracer()
    assert is_compatible_with_meta()
    wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, inps)
    graph = tracer.trace(model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)

    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    offload_strategies_constructor = OffloadStrategiesConstructor(graph)
    offload_strategies_constructor.build_strategies_and_cost()

    solver = Solver(gm.graph, offload_strategies_constructor, memory_budget)
    solver._call_solver_greedy_v1()
    # solver._call_solver_l2l()

    gm = runtime_offload_apply_pass(gm)
    gm.recompile()
    optimized_model = OffloadModuleWrapper(gm)
    return optimized_model


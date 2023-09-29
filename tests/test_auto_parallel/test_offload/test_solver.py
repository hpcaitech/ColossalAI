import pytest
import torch.fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_map

from colossalai.auto_parallel.offload.region_manager import RegionManager
from colossalai.auto_parallel.offload.solver import NOT_NVML, SolverFactory
from colossalai.fx import ColoTracer, is_compatible_with_meta
from colossalai.fx.passes.meta_info_prop import MetaInfoProp
from colossalai.testing import clear_cache_before_run, parameterize
from tests.test_auto_parallel.test_offload.model_utils import *


@pytest.mark.skipif(NOT_NVML, reason="pynvml is not installed")
@clear_cache_before_run()
@parameterize("model_name", ["gpt2_", "bert_"])
@parameterize("memory_budget", [4000])
@parameterize("solver_name", ["syn", "asyn"])
def solver_test(model_name: str, memory_budget: float, solver_name: str):
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    data_args = data_gen(device="cpu")
    wrap_fn = lambda x: x.to(dtype=torch.half) if isinstance(x, torch.Tensor) and torch.is_floating_point(x) else x
    data_args = tree_map(wrap_fn, data_args)
    model = model_builder()
    model.train()
    model = model.cpu().half()

    tracer = ColoTracer()
    assert is_compatible_with_meta()
    wrap_fn = lambda x: x.to("meta") if isinstance(x, torch.Tensor) else x
    meta_args = tree_map(wrap_fn, data_args)
    graph = tracer.trace(model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)

    interp = MetaInfoProp(gm)
    interp.propagate(*meta_args.values())

    region_manager = RegionManager(graph, solver_name=solver_name)
    region_manager._pre_process()
    region_list = region_manager.region_list

    solver_cls = SolverFactory.create(solver_name)
    memory_budget = memory_budget * 1024 * 1024
    solver = solver_cls(region_list, memory_budget)
    solver._call_solver()

    assert solver.best_ts.peak_mem < memory_budget

    print("****************** execution plan *******************")
    for region in region_list:
        need_offload = region.need_offload
        to_prefetch = region.fwd_prefetch_region.r_id if region.fwd_prefetch_region is not None else None
        print(
            f"| {model_name} forward | region id: {region.r_id} | need_offload: {need_offload} | to_prefetch: {to_prefetch}"
        )
    for region in region_list.__reversed__():
        need_offload = region.need_offload
        to_prefetch = region.bwd_prefetch_region.r_id if region.bwd_prefetch_region is not None else None
        print(
            f"| {model_name} backward | region id: {region.r_id} | need_offload: {need_offload} | to_prefetch: {to_prefetch}"
        )


if __name__ == "__main__":
    solver_test()

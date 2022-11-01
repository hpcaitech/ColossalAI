import copy
from pprint import pprint
from typing import Dict, List

import torch
from torch.fx import GraphModule

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.solver import SolverOptions, StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.meta_profiler import MetaInfo, meta_register
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.shape_consistency import to_global
from colossalai.testing.comparison import assert_close, assert_close_loose


def _build_model_to_compare(model: torch.nn.Module, input_args: List[torch.Tensor],
                            input_kwargs: Dict[str, torch.Tensor], grad_dict: Dict[any, torch.Tensor]):

    model_to_compare = copy.deepcopy(model)
    args_to_compare = []
    kwargs_to_compare = {}
    for arg_index, input_tensor in enumerate(input_args):

        def wrapper(param, index):

            def hook_fn(grad):
                grad_dict[index] = grad

            param.register_hook(hook_fn)

        arg_to_compare = copy.deepcopy(input_tensor)
        arg_to_compare.requires_grad = True
        wrapper(arg_to_compare, arg_index)
        args_to_compare.append(arg_to_compare)

    for name, input_kwarg in input_kwargs.items():

        def wrapper(param, name):

            def hook_fn(grad):
                grad_dict[name] = grad

            param.register_hook(hook_fn)

        kwarg_to_compare = copy.deepcopy(input_kwarg)
        kwarg_to_compare.requires_grad = True
        wrapper(kwarg_to_compare, name)
        kwargs_to_compare[name] = kwarg_to_compare

    return model_to_compare, args_to_compare, kwargs_to_compare


def mem_test_for_node_strategy(rank: int,
                               model: torch.nn.Module,
                               device_mesh: DeviceMesh,
                               node_index: int,
                               strategy_number: int,
                               input_args: List[torch.Tensor],
                               meta_arg_names: List[str],
                               input_kwargs: Dict[str, torch.Tensor] = {}):
    for strategy_index in range(strategy_number):
        # We need to copy the model to avoid do backward more than once in same graph
        grad_to_shard_dict = {}
        model_to_shard, args_to_shard, kwargs_to_shard = _build_model_to_compare(model, input_args, input_kwargs,
                                                                                 grad_to_shard_dict)

        tracer = ColoTracer()
        input_sample = {}
        for input_arg, meta_arg_name in zip(input_args, meta_arg_names):
            input_sample[meta_arg_name] = torch.rand(input_arg.shape).to('meta')
        for meta_kwarg_name, input_kwarg in input_kwargs.items():
            input_sample[meta_kwarg_name] = torch.rand(input_kwarg.shape).to('meta')
        graph = tracer.trace(root=model_to_shard, meta_args=input_sample)
        gm = GraphModule(model_to_shard, graph, model_to_shard.__class__.__name__)
        solver_options = SolverOptions(fast=True)
        strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
        strategies_constructor.build_strategies_and_cost()
        target_node = list(graph.nodes)[node_index]

        if rank == 0:
            print("=======================")
            print(f"#strategy_index: {strategy_index}")
            pprint(target_node.strategies_vector[strategy_index])

        # solution construction
        # construct the strategy for the target node
        solution_len = len(strategies_constructor.leaf_strategies)
        solution = [0] * solution_len
        solution[node_index] = strategy_index

        # construct the strategy for the output node
        placeholder_strategy = list(graph.nodes)[-1].strategies_vector[0]
        output_key = next(key for key in target_node.strategies_vector[strategy_index].sharding_specs.keys()
                          if key in placeholder_strategy.sharding_specs)
        placeholder_strategy.sharding_specs[output_key] = target_node.strategies_vector[strategy_index].sharding_specs[
            output_key]

        gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict = runtime_preparation_pass(
            gm, solution, device_mesh)
        gm = runtime_apply_pass(gm)
        gm.recompile()

        # warmup
        with torch.no_grad():
            output = gm(*args_to_shard,
                        sharding_spec_convert_dict=sharding_spec_dict,
                        origin_node_sharding_spec_dict=origin_spec_dict,
                        comm_actions_dict=comm_actions_dict,
                        **kwargs_to_shard)

        del output
        # forward memory compare
        if rank == 0:
            torch.cuda.reset_peak_memory_stats("cuda:0")
            mem_stamp0 = torch.cuda.memory_allocated("cuda:0")
        output = gm(*args_to_shard,
                    sharding_spec_convert_dict=sharding_spec_dict,
                    origin_node_sharding_spec_dict=origin_spec_dict,
                    comm_actions_dict=comm_actions_dict,
                    **kwargs_to_shard)

        if rank == 0:
            # print forward memory allocated and peak memory stats in kb
            print(
                f"forward memory allocated: {(torch.cuda.memory_allocated('cuda:0') - mem_stamp0) / 1024} kb, peak memory stats: {(torch.cuda.max_memory_allocated('cuda:0') - mem_stamp0) / 1024} kb"
            )

        # backward memory compare
        grad_tensors = torch.ones_like(output)
        torch.cuda.reset_peak_memory_stats()
        mem_stamp0 = torch.cuda.memory_allocated()
        torch.autograd.backward(output, grad_tensors)

        if rank == 0:
            # print backward memory allocated and peak memory stats in kb
            print(
                f"backward memory allocated: {(torch.cuda.memory_allocated('cuda:0') - mem_stamp0) / 1024} kb, peak memory stats: {(torch.cuda.max_memory_allocated('cuda:0') - mem_stamp0) / 1024} kb"
            )

            # estimated memory
            metainfo = MetaInfo(target_node.strategies_vector[strategy_index],
                                target_node.graph.owning_module.get_submodule(target_node.target).__class__)
            print("estimated memory:")
            print(
                f"forward activation: {metainfo.memory_cost.fwd.activation / 1024} kb, forward param: {metainfo.memory_cost.fwd.parameter / 1024} kb"
            )
            print(
                f"forward temp: {metainfo.memory_cost.fwd.temp / 1024} kb, forward buffer: {metainfo.memory_cost.fwd.buffer / 1024} kb"
            )
            print(
                f"backward activation: {metainfo.memory_cost.bwd.activation / 1024} kb, backward param: {metainfo.memory_cost.bwd.parameter / 1024} kb"
            )
            print(
                f"backward temp: {metainfo.memory_cost.bwd.temp / 1024} kb, backward buffer: {metainfo.memory_cost.bwd.buffer / 1024} kb"
            )
            print("=======================")

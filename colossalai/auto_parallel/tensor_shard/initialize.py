from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule
from torch.fx.graph import Graph

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.sharding_strategy import CommAction
from colossalai.auto_parallel.tensor_shard.solver import (
    CostGraph,
    GraphAnalyser,
    Solver,
    SolverOptions,
    StrategiesConstructor,
)
from colossalai.device.alpha_beta_profiler import AlphaBetaProfiler
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec


class ModuleWrapper(nn.Module):
    '''
    This class is used to wrap the original module, and add the sharding_spec_dict, origin_spec_dict, comm_actions_dict
    into the forward function.
    '''

    def __init__(self, module: ColoGraphModule, sharding_spec_dict: Dict[int, List[ShardingSpec]],
                 origin_spec_dict: Dict[int, ShardingSpec], comm_actions_dict: Dict[int, Dict[str, CommAction]]):
        '''
        Args:
            module: the original module
            sharding_spec_dict: The sharding_spec_dict is used to record the target sharding specs of each tensor required in user node.
            origin_spec_dict: The origin_spec_dict is used to record the original sharding spec of each tensor.
            comm_actions_dict: The comm_actions_dict is used to record the communication actions of each tensor.
        '''
        super(ModuleWrapper, self).__init__()
        self.module = module
        self.sharding_spec_dict = sharding_spec_dict
        self.origin_spec_dict = origin_spec_dict
        self.comm_actions_dict = comm_actions_dict

    def forward(self, *args, **kwargs):
        return self.module(*args,
                           sharding_spec_convert_dict=self.sharding_spec_dict,
                           origin_node_sharding_spec_dict=self.origin_spec_dict,
                           comm_actions_dict=self.comm_actions_dict,
                           **kwargs)


def extract_meta_args_from_dataloader(data_loader: torch.utils.data.DataLoader, data_process_func: callable):
    '''
    This method is used to extract the meta_args from the dataloader under the instruction of the data_process_func.
    '''
    # TODO: implement this function
    pass


def extract_alpha_beta_for_device_mesh(alpha_beta_dict: Dict[Tuple[int], Tuple[float]], logical_mesh_shape: Tuple[int]):
    '''
    This method is used to extract the mesh_alpha and mesh_beta for the given logical_mesh_shape
    from the alpha_beta_dict. These two values will be used to estimate the communication cost.
    '''
    # TODO: implement this function
    pass


def build_strategy_constructor(graph: Graph, device_mesh: DeviceMesh):
    '''
    This method is used to build the strategy_constructor for the given graph.
    After this method, each node in the graph will have a strategies_vector which
    is constructed by the related node handler.
    '''
    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    return strategies_constructor


def solve_solution(gm: ColoGraphModule, strategy_constructor: StrategiesConstructor, memory_budget: float = -1.0):
    '''
    This method is used to solve the best solution for the given graph.
    The solution is a list of integers, each integer represents the best strategy index of the corresponding node.
    '''
    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    cost_graph = CostGraph(strategy_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategy_constructor, cost_graph, graph_analyser, memory_budget=memory_budget)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])

    return solution


def transform_to_sharded_model(gm: ColoGraphModule, solution: List[int], device_mesh: DeviceMesh,
                               strategies_constructor: StrategiesConstructor):
    '''
    This method is used to transform the original graph to the sharded graph.
    The model parameters will be sharded according to the solution and the grad hooks
    will be added to the sharded graph using the runtime_preparation_pass.
    The communication node will be added into the graph using the runtime_apply_pass.
    '''
    gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict = runtime_preparation_pass(
        gm, solution, device_mesh, strategies_constructor)
    gm = runtime_apply_pass(gm)
    gm.recompile()
    sharding_spec_dicts = (sharding_spec_dict, origin_spec_dict, comm_actions_dict)

    return gm, sharding_spec_dicts


def initialize_device_mesh(world_size: int = -1,
                           physical_devices: List[int] = None,
                           alpha_beta_dict: Dict[Tuple[int], Tuple[float]] = None,
                           logical_mesh_shape: Tuple[int] = None,
                           logical_mesh_id: torch.Tensor = None):
    '''
    This method is used to initialize the device mesh.

    Args:
        world_size: the size of device mesh. If the world_size is -1,
            the world size will be set to the number of GPUs in the current machine.
        physical_devices: the physical devices used to initialize the device mesh.
        alpha_beta_dict(optional): the alpha_beta_dict contains the alpha and beta values
            for each devices. if the alpha_beta_dict is None, the alpha_beta_dict will be
            generated by profile_alpha_beta function.
        logical_mesh_shape(optional): the logical_mesh_shape is used to specify the logical
            mesh shape.
        logical_mesh_id(optional): the logical_mesh_id is used to specify the logical mesh id.
    '''
    # if world_size is not set, use the world size from torch.distributed
    if world_size == -1:
        world_size = dist.get_world_size()

    if physical_devices is None:
        physical_devices = [i for i in range(world_size)]
    physical_mesh = torch.tensor(physical_devices)

    if alpha_beta_dict is None:
        # if alpha_beta_dict is not given, use a series of executions to profile alpha and beta values for each device
        ab_profiler = AlphaBetaProfiler(physical_devices)
        alpha_beta_dict = ab_profiler.alpha_beta_dict
    else:
        ab_profiler = AlphaBetaProfiler(physical_devices, alpha_beta_dict=alpha_beta_dict)

    if logical_mesh_shape is None and logical_mesh_id is None:
        # search for the best logical mesh shape
        logical_mesh_id = ab_profiler.search_best_logical_mesh()
        logical_mesh_id = torch.Tensor(logical_mesh_id).to(torch.int)
        logical_mesh_shape = logical_mesh_id.shape

        # extract alpha and beta values for the chosen logical mesh shape
        mesh_alpha, mesh_beta = ab_profiler.extract_alpha_beta_for_device_mesh()

    elif logical_mesh_shape is not None and logical_mesh_id is None:
        logical_mesh_id = physical_mesh.reshape(logical_mesh_shape)

        # extract alpha and beta values for the chosen logical mesh shape
        mesh_alpha, mesh_beta = extract_alpha_beta_for_device_mesh(alpha_beta_dict, logical_mesh_id)

    device_mesh = DeviceMesh(physical_mesh_id=physical_mesh,
                             logical_mesh_id=logical_mesh_id,
                             mesh_alpha=mesh_alpha,
                             mesh_beta=mesh_beta,
                             init_process_group=True)
    return device_mesh


def initialize_model(model: nn.Module,
                     meta_args: Dict[str, torch.Tensor],
                     device_mesh: DeviceMesh,
                     memory_budget: float = -1.0,
                     save_solver_solution: bool = False,
                     load_solver_solution: bool = False,
                     solution_path: str = None,
                     return_solution: bool = False):
    '''
    This method is used to initialize the sharded model which could be used as normal pytorch model.

    Args:
        model: the model to be sharded.
        meta_args: the meta_args is used to specify the input shapes of the model.
        device_mesh: the device mesh to execute the model.
        memory_budget(optional): the max cuda memory could be used. If the memory budget is -1.0,
            the memory budget will be infinity.
        save_solver_solution(optional): if the save_solver_solution is True, the solution will be saved
            to the solution_path.
        load_solver_solution(optional): if the load_solver_solution is True, the solution will be loaded
            from the solution_path.
        solution_path(optional): the path to save or load the solution.
        return_solution(optional): if the return_solution is True, the solution will be returned. The returned
            solution will be used to debug or help to analyze the sharding result. Therefore, we will not just
            return a series of integers, but return the best strategies.
    '''
    tracer = ColoTracer(trace_act_ckpt=True)

    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    strategies_constructor = build_strategy_constructor(graph, device_mesh)
    if load_solver_solution:
        solution = torch.load(solution_path)
    else:
        solution = solve_solution(gm, strategies_constructor, memory_budget)
        if save_solver_solution:
            torch.save(solution, solution_path)

    gm, sharding_spec_dicts = transform_to_sharded_model(gm, solution, device_mesh, strategies_constructor)
    model_to_return = ModuleWrapper(gm, *sharding_spec_dicts)

    if return_solution:
        solution_to_return = []
        nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
        for index, node in enumerate(nodes):
            solution_to_return.append(f'{node.name} {node.strategies_vector[solution[index]].name}')
        return model_to_return, solution_to_return
    else:
        return model_to_return


def autoparallelize(model: nn.Module,
                    meta_args: Dict[str, torch.Tensor] = None,
                    data_loader: torch.utils.data.DataLoader = None,
                    data_process_func: callable = None,
                    alpha_beta_dict: Dict[Tuple[int], Tuple[float]] = None,
                    logical_mesh_shape: Tuple[int] = None,
                    logical_mesh_id: torch.Tensor = None,
                    save_solver_solution: bool = False,
                    load_solver_solution: bool = False,
                    solver_solution_path: str = None,
                    return_solution: bool = False,
                    memory_budget: float = -1.0):
    '''
    This method is used to initialize the device mesh, extract the meta_args, and
    use them to create a sharded model.

    Args:
        model: the model to be sharded.
        meta_args(optional): the meta_args is used to specify the input shapes of the model.
            If the meta_args is None, the meta_args will be extracted from the data_loader.
        data_loader(optional): the data_loader to be used in normal training loop.
        data_process_func(optional): the data_process_func is used to process the data from the data_loader.
        alpha_beta_dict(optional): the alpha_beta_dict contains the alpha and beta values
            for each devices. if the alpha_beta_dict is None, the alpha_beta_dict will be
            generated by profile_alpha_beta function.
        logical_mesh_shape(optional): the logical_mesh_shape is used to specify the logical
            mesh shape. If the logical_mesh_shape is None, the logical_mesh_shape will be
            generated by search_best_logical_mesh_shape function.
        logical_mesh_id(optional): the logical_mesh_id is used to specify the logical mesh id.
        save_solver_solution(optional): if the save_solver_solution is True, the solution will be saved
            to the solution_path.
        load_solver_solution(optional): if the load_solver_solution is True, the solution will be loaded
            from the solution_path.
        solver_solution_path(optional): the path to save or load the solution.
        return_solution(optional): if the return_solution is True, the solution will be returned.
        memory_budget(optional): the max cuda memory could be used. If the memory budget is -1.0,
            the memory budget will be infinity.
    '''
    device_mesh = initialize_device_mesh(alpha_beta_dict=alpha_beta_dict,
                                         logical_mesh_shape=logical_mesh_shape,
                                         logical_mesh_id=logical_mesh_id)
    if meta_args is None:
        meta_args = extract_meta_args_from_dataloader(data_loader, data_process_func)

    rst_to_unpack = initialize_model(model,
                                     meta_args,
                                     device_mesh,
                                     save_solver_solution=save_solver_solution,
                                     load_solver_solution=load_solver_solution,
                                     solution_path=solver_solution_path,
                                     return_solution=return_solution,
                                     memory_budget=memory_budget)

    if return_solution:
        model, solution = rst_to_unpack
        return model, solution
    else:
        model = rst_to_unpack
        return model

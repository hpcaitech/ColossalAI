import operator
from functools import reduce
import torch
from colossalai.auto_parallel.solver.sharding_strategy import ShardingStrategy, StrategiesVector
from .operator_handler import OperatorHandler


class ConvHandler(OperatorHandler):
    """
    A OperatorHandler which deals with the sharding strategies of linear matrix multiplication.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = self.predecessor_node[0]._meta_data
        self.weight = self.module_named_parameters['weight']
        self.output_data = self.node._meta_data
        self._sanity_check()

    def _sanity_check(self):
        '''
        In sanity check, we need make sure the input data having correct dimension size.
        For Conv1d, the dim of input data should be 3([N, C, L]).
        For Conv2d, the dim of input data should be 4([N, C, H, W]).
        For Conv3d, the dim of input data should be 5([N, C, H, W, D]).
        '''
        assert self.input_data.dim() in (3, 4,
                                         5), f'We suppose the dim of input fed into conv op should in range of [3, 5].'

    def _generate_compute_cost(self, bs, channel_in, channel_out):
        '''
        Compute the computation cost per device with this specific strategy.

        Note: compute_cost need to be devided by TFLOPS, now it just shows the computation size.

        Argument:
            bs(int): Batch size of the input data.
            channel_in(int): The channel dimension of input data.
            channel_out(int): The out channel of the conv weight.

        Return:
            compute_cost(float): Computation cost per device with this specific strategy
        '''
        # TODO: compute_cost need to be devided by TFLOPS, now it just shows the computation size.
        # 1D: (L) * N * Cout * Cin * kernel
        # 2D: (H * W) * N * Cout * Cin * kernel
        # 3D: (H * W  * D) * N * Cout * Cin * kernel
        output_size = self.output_data.shape[2:]
        output_size_product = reduce(operator.mul, output_size, 1)
        kernel_size = self.weight.shape[2:]
        kernel_size_product = reduce(operator.mul, kernel_size, 1)
        compute_cost = output_size_product * bs * channel_in * channel_out * kernel_size_product
        return compute_cost

    def split_input_batch_weight_out_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}S{mesh_dim_1} = S{mesh_dim_0}R x RS{mesh_dim_1}'

        dim_partition_dict_for_input = {0: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_output)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1] // self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0] * self.device_mesh.shape[mesh_dim_1]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # This strategy do not need to do all_reduce operation
        communication_cost = 0
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def split_input_both_dim_weight_in_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'S{mesh_dim_0}R = S{mesh_dim_0}S{mesh_dim_1} x S{mesh_dim_1}R'

        dim_partition_dict_for_input = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {0: [mesh_dim_0]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0] // self.device_mesh.shape[mesh_dim_0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_1]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def split_input_in_channel_weight_both_channel(self, mesh_dim_0, mesh_dim_1):
        name = f'RS{mesh_dim_1} = RS{mesh_dim_0} x S{mesh_dim_0}S{mesh_dim_1}'

        dim_partition_dict_for_input = {1: [mesh_dim_0]}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {0: [mesh_dim_0], 1: [mesh_dim_1]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_1]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1] // self.device_mesh.shape[mesh_dim_0]
        channel_out = self.weight.shape[1] // self.device_mesh.shape[mesh_dim_1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # compute the communication cost of this strategy
        communication_cost = self.device_mesh.all_reduce_cost(memory_cost, mesh_dim_1)
        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def split_weight_out_channel(self, mesh_dim_0):
        name = f'RS{mesh_dim_0} = RR x RS{mesh_dim_0}'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {1: [mesh_dim_0]}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {1: [mesh_dim_0]}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1] // self.device_mesh.shape[mesh_dim_0]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        sharding_size = self.device_mesh.shape[mesh_dim_0]
        memory_cost = numel * size_per_elem_bytes / sharding_size

        # This strategy do not need to do all_reduce operation
        communication_cost = 0

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def non_split(self):
        name = f'RR = RR x RR'

        dim_partition_dict_for_input = {}
        sharding_spec_for_input = self._generate_sharding_spec(self.input_data, dim_partition_dict_for_input)

        dim_partition_dict_for_weight = {}
        sharding_spec_for_weight = self._generate_sharding_spec(self.weight, dim_partition_dict_for_weight)

        dim_partition_dict_for_output = {}
        sharding_spec_for_ouput = self._generate_sharding_spec(self.output_data, dim_partition_dict_for_input)

        # generate resharding cost for this strategy
        resharding_costs = self._generate_resharding_costs([sharding_spec_for_input])

        # compute the computation cost of this strategy
        bs = self.input_data.shape[0]
        channel_in = self.input_data.shape[1]
        channel_out = self.weight.shape[1]
        compute_cost = self._generate_compute_cost(bs, channel_in, channel_out)

        # compute the memory cost of this strategy
        dtype = self.input_data.dtype
        numel = self.output_data.numel()
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
        memory_cost = numel * size_per_elem_bytes

        # This strategy do not need to do all_reduce operation
        communication_cost = 0

        sharding_strategies = ShardingStrategy(name,
                                               output_sharding_spec=sharding_spec_for_ouput,
                                               compute_cost=compute_cost,
                                               communication_cost=communication_cost,
                                               memory_cost=memory_cost,
                                               resharding_costs=resharding_costs,
                                               input_shardings=(sharding_spec_for_input, sharding_spec_for_weight))
        self.strategies_vector.append(sharding_strategies)

    def register_strategy(self) -> StrategiesVector:
        '''
        Generate every possible strategies for a Conv node, and record all strategies into the strategies_vector.

        Example:
            physical_mesh_id = torch.arange(0, 4)
            mesh_shape = (2, 2)
            # [[0, 1]
            #  [2, 3]]
            device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
            shape_consistency_manager = ShapeConsistencyManager()

            model = ConvModel(16, 32)
            input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}
            # graph():
            #     %x : torch.Tensor [#users=1] = placeholder[target=x]
            #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
            #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
            #     return conv
            graph = tracer.trace(root=model, meta_args=input_sample)
            gm = GraphModule(model, graph, model.__class__.__name__)
            gm.recompile()
            # [x, mul, conv, output]
            nodes = [node for node in gm.graph.nodes]

            # strategies_for_input = [[R, R, R, R], [R, S0, R, R], [R, S1, R, R], [S0, R, R, R], [S0, S1, R, R], [S1, R, R, R], [S1, S0, R, R]]
            strategies_vector_for_input = StrategiesVector(node=nodes[0], in_nodes=[nodes[1], 2], strategies=strategies_for_input)
            setattr(nodes[1], 'strategies_vector', strategies_vector_for_input)
            
            strategies_vector = StrategiesVector(node=nodes[2], in_nodes=[nodes[1], ])
            conv_handler = ConvHandler(input_node=nodes[1], input_index=0, weight=dict(gm.named_modules())[nodes[2].name].weight, output_node=nodes[2],
                                    device_mesh=device_mesh, strategies_vector=strategies_vector, shape_consistency_manager=shape_consistency_manager)
            conv_handler.register_strategy_into_strategies_vector()
            for strategy in conv_handler.strategies_vector.strategies:
                print(f'{strategy.name}: compute_cost is {strategy.compute_cost}, communication_cost is {strategy.communication_cost}, memory_cost is {strategy.memory_cost}, resharding_costs is {strategy.resharding_costs}')
        
        Output:
            S0S1 = S0R x RS1: compute_cost is 8856576, communication_cost is 0, memory_cost is 492032.0, resharding_costs is {0: [0, 32769.001, 131074.2, 0, 32769.1, 131074.2, 98307.201]}
            S1S0 = S1R x RS0: compute_cost is 8856576, communication_cost is 0, memory_cost is 492032.0, resharding_costs is {0: [0, 131074.2, 32769.001, 131074.2, 98307.201, 0, 32769.1]}
            S0R = S0S1 x S1R: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {0: [0, 65538.002, 0, 0, 0, 65538.002, 196614.402]}
            S1R = S1S0 x S0R: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {0: [0, 0, 65538.002, 65538.002, 196614.402, 0, 0]}
            RS1 = RS0 x S0S1: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {0: [0, 0, 131074.2, 32769.001, 98307.201, 131074.2, 32769.1]}
            RS0 = RS1 x S1S0: compute_cost is 8856576, communication_cost is 984065.01, memory_cost is 984064.0, resharding_costs is {0: [0, 131074.2, 0, 131074.2, 32769.1, 32769.001, 98307.201]}
            RS0 = RR x RS0: compute_cost is 17713152, communication_cost is 0, memory_cost is 984064.0, resharding_costs is {0: [0, 65537.1, 65537.1, 65537.1, 131075.30000000002, 65537.1, 131075.30000000002]}
            RS1 = RR x RS1: compute_cost is 17713152, communication_cost is 0, memory_cost is 984064.0, resharding_costs is {0: [0, 65537.1, 65537.1, 65537.1, 131075.30000000002, 65537.1, 131075.30000000002]}
            RR = RR x RR: compute_cost is 35426304, communication_cost is 0, memory_cost is 1968128, resharding_costs is {0: [0, 65537.1, 65537.1, 65537.1, 131075.30000000002, 65537.1, 131075.30000000002]}
        '''
        # SS = SR x RS
        self.split_input_batch_weight_out_channel(0, 1)
        self.split_input_batch_weight_out_channel(1, 0)

        # SR = SS x SR
        self.split_input_both_dim_weight_in_channel(0, 1)
        self.split_input_both_dim_weight_in_channel(1, 0)

        # RS = RS x SS
        self.split_input_in_channel_weight_both_channel(0, 1)
        self.split_input_in_channel_weight_both_channel(1, 0)

        # RS = RR x RS
        self.split_weight_out_channel(0)
        self.split_weight_out_channel(1)

        # RR= RR x RR
        self.non_split()

        return self.strategies_vector

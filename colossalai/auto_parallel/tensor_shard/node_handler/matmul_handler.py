import operator
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from functools import reduce
from typing import Dict, List, Union

import torch

from colossalai.auto_parallel.tensor_shard.utils.broadcast import (
    BroadcastType,
    get_broadcast_dim_info,
    get_broadcast_shape,
)
from colossalai.tensor.sharding_spec import ShardingSpecException

from ..sharding_strategy import OperationData, OperationDataType, ShardingStrategy
from ..utils import recover_sharding_spec_for_broadcast_shape
from .node_handler import MetaInfoNodeHandler
from .registry import operator_registry
from .strategy import (
    BatchedMatMulStrategyGenerator,
    DotProductStrategyGenerator,
    LinearProjectionStrategyGenerator,
    MatVecStrategyGenerator,
    StrategyGenerator,
)


class MatMulType(Enum):
    """
    The MatMulType is categorized into 4 types based on the reference of torch.matmul
    in https://pytorch.org/docs/stable/generated/torch.matmul.html.

    DOT: dot product, both tensors are 1D, these two tensors need to have the same number of elements
    MM: matrix-matrix product, both tensors are 2D or the 1st tensor is 1D and the 2nd tensor is 2D
    MV: matrix-vector product: the 1st tensor is 2D and the 2nd tensor is 1D
    BMM: batched matrix-matrix multiplication, one tensor is at least 1D and the other is at least 3D
    """

    DOT = 0
    MM = 1
    MV = 2
    BMM = 3


def get_matmul_type(input_dim: int, other_dim: int):
    """
    Determine which type of matmul operation should be executed for the given tensor dimensions.

    Args:
        input_dim (int): the number of dimensions for the input tensor
        other_dim (int): the number of dimensions for the other tensor
    """
    if input_dim == 1 and other_dim == 1:
        matmul_type = MatMulType.DOT
    elif input_dim in [1, 2] and other_dim == 2:
        matmul_type = MatMulType.MM
    elif input_dim == 2 and other_dim == 1:
        matmul_type = MatMulType.MV
    elif input_dim >= 1 and other_dim >= 1 and (input_dim > 2 or other_dim > 2):
        matmul_type = MatMulType.BMM
    else:
        raise ValueError(
            f"The input and other tensors are of {input_dim} and {other_dim} which cannot used to execute matmul operation"
        )
    return matmul_type


class BmmTransform(ABC):
    """
    BmmTransform is an abstraction of the shape conversion between logical and physical operation data
    during the strategy generation.
    """

    @abstractmethod
    def apply(self, shape_mapping: Dict[str, List[int]]):
        pass

    @abstractmethod
    def recover(self, op_data_mapping: Dict[str, OperationData], strategy: ShardingStrategy):
        pass


class Padder(BmmTransform):
    """
    Add padding to the matrix dimensions for batched matrix multiplication.
    """

    def __init__(self) -> None:
        # keep the padding dim, op_name -> padded_dim
        self.padded_dim_mapping = {}

    def apply(self, shape_mapping: Dict[str, List[int]]):
        mapping_copy = deepcopy(shape_mapping)
        input_shape = mapping_copy["input"]
        other_shape = mapping_copy["other"]

        if len(input_shape) == 1:
            # if the input is a 1D tensor, 1 is prepended to its shape
            # and it will be removed afterwards
            input_shape.insert(0, 1)
            self.padded_dim_mapping["input"] = -2
            self.padded_dim_mapping["output"] = -2
        elif len(other_shape) == 1:
            # if the other is a 1D tensor, 1 is appended to its shape
            # and it will be removed afterwards
            other_shape = other_shape.append(1)
            self.padded_dim_mapping["other"] = -1
            self.padded_dim_mapping["output"] = -1
        return mapping_copy

    def recover(self, op_data_mapping: Dict[str, OperationData], strategy: ShardingStrategy):
        op_data_mapping["input"]
        op_data_mapping["other"]

        def _remove_padded_dim(key, strategy):
            op_data = op_data_mapping[key]
            sharding_spec = strategy.get_sharding_spec_by_name(op_data.name)
            tensor_shape = list(sharding_spec.entire_shape)
            dim_partition_list = [None] * len(tensor_shape)

            # padded dim is a negative number as the padded dim must be a matrix dim
            padded_dim = self.padded_dim_mapping[key]

            # compute the new dim partition
            for tensor_dim, mesh_dims in sharding_spec.dim_partition_dict.items():
                dim_partition_list[tensor_dim] = mesh_dims
            dim_partition_list.pop(padded_dim)
            unpadded_dim_partition_list = {k: v for k, v in enumerate(dim_partition_list) if v is not None}

            # compute unpadded tensor shape
            tensor_shape.pop(padded_dim)

            assert tensor_shape == list(op_data.data.shape), f"{tensor_shape} vs {list(op_data.data.shape)}"

            # update sharding spec
            sharding_spec.__init__(sharding_spec.device_mesh, tensor_shape, unpadded_dim_partition_list)

        # enumerate all sharding strategies
        strategies = []
        try:
            strategy_copy = strategy.clone()

            # only one of input and other will be padded
            if "input" in self.padded_dim_mapping:
                _remove_padded_dim("input", strategy_copy)
                _remove_padded_dim("output", strategy_copy)
            elif "other" in self.padded_dim_mapping:
                _remove_padded_dim("other", strategy_copy)
                _remove_padded_dim("output", strategy_copy)

            strategies.append(strategy_copy)
        except ShardingSpecException:
            pass
        return strategies


class Broadcaster(BmmTransform):
    """
    Broadcast the non-matrix dimensions for batched matrix multiplication.
    """

    def __init__(self) -> None:
        self.broadcast_dim_info = {}

    def apply(self, shape_mapping: Dict[str, List[int]]):
        mapping_copy = shape_mapping.copy()

        # get shapes
        input_shape = mapping_copy["input"]
        other_shape = mapping_copy["other"]

        # sanity check
        assert len(input_shape) > 1 and len(other_shape) > 1

        # broadcast the batch dim and record
        bcast_non_matrix_dims = get_broadcast_shape(input_shape[:-2], other_shape[:-2])

        # store the broadcast dim info
        input_broadcast_dim_info = get_broadcast_dim_info(bcast_non_matrix_dims, input_shape[:-2])
        other_broadcast_dim_info = get_broadcast_dim_info(bcast_non_matrix_dims, other_shape[:-2])
        self.broadcast_dim_info["input"] = input_broadcast_dim_info
        self.broadcast_dim_info["other"] = other_broadcast_dim_info

        # create the full logical shape
        input_shape = bcast_non_matrix_dims + input_shape[-2:]
        other_shape = bcast_non_matrix_dims + other_shape[-2:]
        assert len(input_shape) == len(other_shape)

        mapping_copy["input"] = input_shape
        mapping_copy["other"] = other_shape

        return mapping_copy

    def recover(self, op_data_mapping: Dict[str, OperationData], strategy: ShardingStrategy):
        # remove sharding on the broadcast dim
        def _remove_sharding_on_broadcast_dim(key, strategy):
            op_data = op_data_mapping[key]
            sharding_spec = strategy.get_sharding_spec_by_name(op_data.name)
            tensor_shape = list(sharding_spec.entire_shape)

            for dim_idx, broadcast_type in self.broadcast_dim_info[key].items():
                if broadcast_type == BroadcastType.MULTIPLE:
                    # if the dim is originally 1 and multiplied during broadcast
                    # we set its sharding to R
                    # e.g. [1, 2, 4] x [4, 4, 8] -> [4, 2, 8]
                    # the dim 0 of [1, 2, 4] is multiplied to 4
                    tensor_shape[dim_idx] = 1
                elif broadcast_type == BroadcastType.PADDING:
                    # if the dim is padded
                    # we remove its sharding
                    tensor_shape[dim_idx] = None

            tensor_shape_before_broadcast = [dim for dim in tensor_shape if dim is not None]

            physical_sharding_spec, removed_dims = recover_sharding_spec_for_broadcast_shape(
                logical_sharding_spec=sharding_spec,
                logical_shape=sharding_spec.entire_shape,
                physical_shape=tensor_shape_before_broadcast,
            )
            strategy.sharding_specs[op_data] = physical_sharding_spec

        # enumerate all sharding strategies
        strategies = []
        try:
            strategy_copy = strategy.clone()
            _remove_sharding_on_broadcast_dim("input", strategy_copy)
            _remove_sharding_on_broadcast_dim("other", strategy_copy)
            strategies.append(strategy_copy)
        except ShardingSpecException:
            pass
        return strategies


class Viewer(BmmTransform):
    """
    Change the shape of the tensor from N-D to 3D
    """

    def __init__(self) -> None:
        self.batch_dims_before_view = None

    def apply(self, shape_mapping: Dict[str, List[int]]):
        mapping_copy = shape_mapping.copy()
        self.batch_dims_before_view = list(mapping_copy["input"][:-2])

        # get shapes
        input_shape = shape_mapping["input"]
        other_shape = shape_mapping["other"]

        # view to 3d tensor
        assert len(input_shape) >= 3 and len(other_shape) >= 3
        input_shape = [reduce(operator.mul, input_shape[:-2])] + input_shape[-2:]
        other_shape = [reduce(operator.mul, other_shape[:-2])] + other_shape[-2:]
        output_shape = input_shape[:2] + other_shape[2:]
        mapping_copy["input"] = input_shape
        mapping_copy["other"] = other_shape
        mapping_copy["output"] = output_shape
        return mapping_copy

    def recover(self, op_data_mapping: Dict[str, OperationData], strategy: ShardingStrategy):
        # get operation data
        def _update_sharding_spec(key, strategy, physical_batch_dim):
            """
            Map the logical batch dim to the physical batch dim
            """
            op_data = op_data_mapping[key]
            sharding_spec = strategy.get_sharding_spec_by_name(op_data.name)
            dim_partition_dict = sharding_spec.dim_partition_dict
            entire_shape = sharding_spec.entire_shape

            # update the dimension index for the matrix dimensions
            if 2 in dim_partition_dict:
                dim_partition_dict[len(self.batch_dims_before_view) + 1] = dim_partition_dict.pop(2)
            if 1 in dim_partition_dict:
                dim_partition_dict[len(self.batch_dims_before_view)] = dim_partition_dict.pop(1)

            # map the logical batch dim to physical batch dim
            if 0 in dim_partition_dict:
                batch_dim_shard = dim_partition_dict.pop(0)
                dim_partition_dict[physical_batch_dim] = batch_dim_shard

            # the new shape will be the batch dims + the last 2 matrix dims
            shape_before_view = self.batch_dims_before_view + list(entire_shape[-2:])
            sharding_spec.__init__(sharding_spec.device_mesh, shape_before_view, dim_partition_dict)

        num_batch_dim_before_view = len(self.batch_dims_before_view)

        # enumerate all sharding strategies
        strategies = []
        for i in range(num_batch_dim_before_view):
            # create a new strategy
            strategy_copy = strategy.clone()
            try:
                _update_sharding_spec("input", strategy_copy, i)
                _update_sharding_spec("other", strategy_copy, i)
                _update_sharding_spec("output", strategy_copy, i)
                strategies.append(strategy_copy)
            except ShardingSpecException:
                continue
        return strategies


def _get_bmm_logical_shape(input_shape, other_shape, transforms):
    """
    Compute the logical shapes for BMM operation. BMM has a general representation
    [b, i, k] = [b, i, j] x [b, j, k]

    The dimension b is called non-matrix (batch) dimension and the remaining dimensions are called matrix dimensions
    The logical shape for the bmm operands will undergo three stages
        1. append/prepend the 1 to the 1D tensor if there is any
        2. broadcast the non-matrix dimensions
        3. reshape to 3 dimensions

    """
    shape_mapping = {"input": input_shape, "other": other_shape}

    for transform in transforms:
        shape_mapping = transform.apply(shape_mapping)

    input_shape = shape_mapping.get("input", None)
    other_shape = shape_mapping.get("other", None)
    output_shape = shape_mapping.get("output", None)

    return input_shape, other_shape, output_shape


@operator_registry.register(torch.matmul)
@operator_registry.register(torch.Tensor.matmul)
class MatMulHandler(MetaInfoNodeHandler):
    """
    The MatMulHandler is a node handler which handles the sharding strategy generation for the matmul operation.
    According to https://pytorch.org/docs/stable/generated/torch.matmul.html, the operations will vary depending on
    the operands.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # check which type of operation this matmul will call
        self.input_meta_data = self.node.args[0]._meta_data
        self.other_meta_data = self.node.args[1]._meta_data
        self.output_meta_data = self.node._meta_data

        input_dim = self.input_meta_data.dim()
        other_dim = self.other_meta_data.dim()
        self.matmul_type = get_matmul_type(input_dim, other_dim)

        if self.matmul_type == MatMulType.BMM:
            # bmm operation can possibly involve padding, broadcasting and view
            # these transforms will be used to create logical shape and
            # recover physical sharding spec
            self.transforms = [Padder(), Broadcaster(), Viewer()]
        else:
            self.transforms = None

    def get_strategy_generator(self) -> List[StrategyGenerator]:
        generators = []
        op_data_mapping = self.get_operation_data_mapping()
        if self.matmul_type == MatMulType.BMM:
            generators.append(BatchedMatMulStrategyGenerator(op_data_mapping, self.device_mesh))
        elif self.matmul_type == MatMulType.DOT:
            generators.append(DotProductStrategyGenerator(op_data_mapping, self.device_mesh))
        elif self.matmul_type == MatMulType.MV:
            generators.append(MatVecStrategyGenerator(op_data_mapping, self.device_mesh))
        elif self.matmul_type == MatMulType.MM:
            generators.append(
                LinearProjectionStrategyGenerator(op_data_mapping, self.device_mesh, linear_projection_type="linear")
            )
        return generators

    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        logical_shape_func = {
            MatMulType.DOT: self._get_logical_shape_for_dot,
            MatMulType.MM: self._get_logical_shape_for_mm,
            MatMulType.MV: self._get_logical_shape_for_mv,
            MatMulType.BMM: self._get_logical_shape_for_bmm,
        }
        logical_shapes = logical_shape_func[self.matmul_type]()
        op_data_mapping = self._get_op_data_mapping(*logical_shapes)
        return op_data_mapping

    def _get_op_data_mapping(self, input_logical_shape, other_logical_shape, output_logical_shape):
        # convert list to torch.Size
        if input_logical_shape:
            input_logical_shape = torch.Size(input_logical_shape)

        if other_logical_shape:
            other_logical_shape = torch.Size(other_logical_shape)

        if output_logical_shape:
            output_logical_shape = torch.Size(output_logical_shape)

        # create op data
        input_op_data = OperationData(
            name=str(self.node.args[0]),
            type=OperationDataType.ARG,
            data=self.input_meta_data,
            logical_shape=input_logical_shape,
        )
        other_op_data = OperationData(
            name=str(self.node.args[1]),
            type=OperationDataType.ARG,
            data=self.other_meta_data,
            logical_shape=other_logical_shape,
        )
        output_op_data = OperationData(
            name=str(self.node),
            type=OperationDataType.OUTPUT,
            data=self.output_meta_data,
            logical_shape=output_logical_shape,
        )

        mapping = {"input": input_op_data, "other": other_op_data, "output": output_op_data}
        return mapping

    def _get_logical_shape_for_dot(self):
        """
        The operands for the dot operation have the same logical shape as the physical shape
        """
        return None, None, None

    def _get_logical_shape_for_mm(self):
        """
        We need to handle the input tensor for a matrix-matrix multiplication as the input
        tensor can be a 1D or 2D tensor. If it is a 1D tensor, 1 will be prepended to its shape
        (e.g. [4] -> [1, 4]).
        """
        if self.input_meta_data.dim() == 1:
            input_logical_shape = [1] + list(self.input_meta_data.shape)
            input_logical_shape = torch.Size(input_logical_shape)
        else:
            input_logical_shape = None
        return input_logical_shape, None, None

    def _get_logical_shape_for_mv(self):
        """
        No broadcasting or dim insertion occurs for matrix-vector operation.
        """
        return None, None, None

    def _get_logical_shape_for_bmm(self):
        input_physical_shape = list(self.input_meta_data.shape)
        other_physical_shape = list(self.other_meta_data.shape)
        return _get_bmm_logical_shape(input_physical_shape, other_physical_shape, self.transforms)

    def post_process(self, strategy: ShardingStrategy) -> Union[ShardingStrategy, List[ShardingStrategy]]:
        if self.matmul_type in [MatMulType.DOT, MatMulType.MV]:
            return strategy
        elif self.matmul_type == MatMulType.MM:
            if self.input_meta_data.dim() == 1:
                # if a 1 is prepended to the input shape (this occurs when input is a 1D tensor)
                # we need to remove that dim
                input_sharding_spec = strategy.get_sharding_spec_by_name(str(self.node.args[0]))
                input_physical_shape = self.node.args[0]._meta_data.shape
                dim_partition_dict = input_sharding_spec.dim_partition_dict

                # remove the partitioning in the dim 0
                if 0 in dim_partition_dict:
                    dim_partition_dict.pop(0, None)

                # move the partitioning in dim 1 to dim 0
                if -1 in dim_partition_dict:
                    shard = dim_partition_dict.pop(-1)
                    dim_partition_dict[0] = shard
                if 1 in dim_partition_dict:
                    shard = dim_partition_dict.pop(1)
                    dim_partition_dict[0] = shard

                # re-init the sharding spec
                input_sharding_spec.__init__(
                    input_sharding_spec.device_mesh,
                    entire_shape=input_physical_shape,
                    dim_partition_dict=dim_partition_dict,
                )
                return strategy
            else:
                return strategy
        elif self.matmul_type == MatMulType.BMM:
            op_data_mapping = self.get_operation_data_mapping()

            strategies = [strategy]
            # recover the physical sharding spec
            for transform in self.transforms[::-1]:
                recovered_stragies = []
                for strategy_ in strategies:
                    output = transform.recover(op_data_mapping, strategy_)
                    if isinstance(output, ShardingStrategy):
                        recovered_stragies.append(output)
                    elif isinstance(output, (list, tuple)):
                        recovered_stragies.extend(output)
                    else:
                        raise TypeError(
                            f"Found unexpected output type {type(output)} from the recover method of BmmTransform"
                        )
                strategies = recovered_stragies
            for index, strategies in enumerate(strategies):
                strategies.name = f"{strategies.name}_{index}"
            return strategies

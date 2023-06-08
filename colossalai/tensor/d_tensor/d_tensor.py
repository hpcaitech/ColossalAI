from typing import Optional

import torch
from torch.utils._pytree import tree_map

from colossalai.device.device_mesh import DeviceMesh

from .layout import Layout
from .layout_converter import LayoutConverter, to_global
from .sharding_spec import ShardingSpec

__all__ = ['DTensor', 'distribute_tensor', 'distribute_module', 'construct_default_sharding_spec']

layout_converter = LayoutConverter()


class DTensor(torch.Tensor):
    """
    DTensor stands for distributed tensor. It is a subclass of `torch.Tensor` and contains meta information
    about the tensor distribution. The meta information includes the device mesh, the sharding specification,
    and the entire shape of the tensor.

    During runtime, we will not directly use the DTensor objects for computation. Instead, we will only use the
    `DTensor.local_tensor` for computation. The `DTensor.local_tensor` is the local tensor in the current rank.
    In this way, all tensors involved in computation will only be native PyTorch tensors.

    Example:
        ```python
        from colossalai.device import DeviceMesh

        # define your device mesh
        # assume you have 4 GPUs
        physical_mesh_id = torch.arange(0, 4).reshape(1, 4)
        mesh_shape = (2, 2)
        device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

        # define a tensor
        x = torch.rand(16, 32)

        # create sharding spec for the tensor
        # assume the sharding spec is [S, R]
        dim_partition_dict = {
            0: 1
        }
        sharding_spec = ShardingSpec(a.dim(), dim_partition_dict)

        # create a distributed tensor
        d_tensor = DTensor(x, device_mesh, sharding_spec)
        ```

    Args:
        tensor (`torch.Tensor`): the unsharded tensor.
        device_mesh (`DeviceMesh`): the device mesh for abstraction of the compute devices.
        sharding_spec (`ShardingSpec`): the sharding specification which describes how the tensor will be sharded.
    """

    def __init__(self, tensor: torch.Tensor, device_mesh: DeviceMesh, sharding_spec: ShardingSpec):
        # ensure this tensor is not a DTensor
        assert not isinstance(tensor, DTensor), 'The input tensor should not be a DTensor.'

        # store meta info
        self.local_tensor = tensor
        self.data_type = tensor.dtype
        self.global_shape = tensor.shape

        # create distributed layout
        dist_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec, global_shape=self.global_shape)
        self.dist_layout = dist_layout

        # shard the tensor
        self._apply_layout()

    @staticmethod
    def __new__(cls, tensor, *args, **kwargs):
        return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

    def __repr__(self):
        return f"DTensor(\n{self.to_global()}\n{self.dist_layout}"

    def __str__(self):
        return self.__repr__()

    def layout_convert(self, device_mesh: DeviceMesh, sharding_spec: ShardingSpec) -> None:
        '''
        Convert the layout of the tensor from source_spec to target_spec.
        This will update the `local_tensor` and `dist_layout` in place.

        Args:
            target_layout (Layout): the target layout specification.
        '''
        target_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec, global_shape=self.global_shape)
        self.local_tensor = layout_converter.apply(tensor=self.local_tensor,
                                                   source_layout=self.dist_layout,
                                                   target_layout=target_layout)
        self.dist_layout = target_layout

    def _apply_layout(self):
        '''
        Apply the layout to the local tensor during initializing process.
        '''
        # layout converter requires a source and target laytout
        # we construct the source layer for an unsharded tensor
        # and use self.dist_layer as the targer layout for the sharded tensor
        source_spec = construct_default_sharding_spec(self.local_tensor)
        source_layout = Layout(device_mesh=self.dist_layout.device_mesh,
                               sharding_spec=source_spec,
                               global_shape=self.global_shape)
        self.local_tensor = layout_converter.apply(tensor=self.local_tensor,
                                                   source_layout=source_layout,
                                                   target_layout=self.dist_layout)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # convert all DTensors to native pytorch tensors
        # so that operations will be conducted on native tensors
        def filter_arg(arg):
            if isinstance(arg, DTensor):
                return arg.local_tensor
            else:
                return arg

        args = tree_map(filter_arg, args)
        kwargs = tree_map(filter_arg, kwargs)

        # NOTE: if we want to convert the result into DTensor, we need to infer the layout of result from the layout of input tensors
        # and op type.
        return func(*args, **kwargs)

    @property
    def device_mesh(self):
        '''
        Return the device mesh of the tensor.
        '''
        return self.dist_layout.device_mesh

    @property
    def sharding_spec(self):
        '''
        Return the sharding specification of the tensor.
        '''
        return self.dist_layout.sharding_spec

    def to(self, *args, **kwargs):
        '''
        Move the tensor to a new device or convert the tensor to a new dtype.
        '''
        self.local_tensor = self.local_tensor.to(*args, **kwargs)
        self.data_type = self.local_tensor.dtype
        # TODO: update the device mesh process groups or we should just cache
        # both the cpu process groups and the cuda process groups?
        return self

    def to_local(self):
        '''
        Return the local tensor in this rank.
        '''
        return self.local_tensor

    def to_global(self):
        '''
        Recover the global tensor from the distributed tensor by returning a new `torch.Tensor` object.

        Note: This function will all_gather the local tensor to the global tensor and it
        will not change the layout of the DTensor. This function is mainly used for debugging or
        check the correctness of the distributed tensor.
        '''
        return to_global(self.local_tensor, self.dist_layout)


def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh, sharding_spec: ShardingSpec) -> DTensor:
    '''
    Distribute the local tensor to the distributed tensor according to the dist_layout specified.

    Args:
        tensor (`torch.Tensor`): tensor to be distributed.
        device_mesh (`DeviceMesh`): the device mesh for abstraction of the compute devices.
        sharding_spec (`ShardingSpec`): the sharding specification which describes how the tensor will be sharded.

    Returns:
        A 'DTensor' object.
    '''
    return DTensor(tensor, device_mesh, sharding_spec)


def distribute_module(module: torch.nn.Module, partition_fn: Optional[callable] = None) -> torch.nn.Module:
    '''
    This function converts all the parameters in the module to DTensor(DParam).

    Args:
        module (`torch.nn.Module`): the module to be distributed.
        partition_fn (callable): the partition function which will be used to partition the parameters.

    Note: This function is subject to future change as the DParam has not been implemented yet.
    '''
    for name, param in module.named_parameters():
        if param is not None and not isinstance(param, DTensor):
            # TODO: we could convert the parameter to DParam here,
            # the type of the parameter could be an optional argument.
            setattr(module, name, torch.nn.Parameter(partition_fn(name, param.data)))
    return module


def construct_default_sharding_spec(tensor: torch.Tensor,) -> ShardingSpec:
    '''
    Construct the default sharding specification for the tensor.

    Args:
        tensor (`torch.Tensor`): the tensor to be sharded.

    Returns:
        A `ShardingSpec` object without any sharding specified.
    '''
    return ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={})

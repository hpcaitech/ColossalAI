import copy
import operator
from functools import reduce
from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor.sharding_spec import DimSpec

from .layout import Layout
from .layout_converter import LayoutConverter
from .sharding_spec import ShardingSpec

layout_converter = LayoutConverter()

_SHARD_DIM = DimSpec([0])


def get_shard_dim_1d(p: torch.Tensor):
    """
    Get the dimension along which the tensor is sharded, for example in 1D Tensor Parallel.
    Args:
        p (torch.Tensor): the input tensor
    Returns:
        int: the dimension along which the tensor is sharded
    """
    if not is_distributed_tensor(p):
        raise ValueError("p is not a distributed tensor")
    sharding = p.dist_layout.sharding_spec.sharding_sequence
    return sharding.index(_SHARD_DIM)


def clear_layout_converter():
    global layout_converter
    layout_converter.cached_solution.clear()


def is_distributed_tensor(tensor: torch.Tensor) -> bool:
    """
    Check whether the given tensor is a distributed tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a distributed tensor.
    """
    return hasattr(tensor, "dist_layout")


def is_sharded(dtensor: torch.Tensor) -> bool:
    """
    Check if a tensor is sharded.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: True if the tensor is sharded, False otherwise.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    return list(dtensor.shape) == list(dtensor.dist_layout.global_shape)


def _hijack_detach_and_clone(dtensor: torch.Tensor) -> torch.Tensor:
    """
    Hijack the detach and clone methods of the tensor to make sure the dist_layout is copied.

    Args:
        tensor (torch.Tensor): The tensor to be hijacked.

    Returns:
        torch.Tensor: The hijacked tensor.
    """
    dtensor._old_detach = dtensor.detach
    dtensor._old_clone = dtensor.clone

    def new_detach(self):
        t_ = self._old_detach()
        t_.dist_layout = copy.deepcopy(self.dist_layout)
        return t_

    def new_clone(self, *args, **kwargs):
        t_ = self._old_clone(*args, **kwargs)
        t_.dist_layout = copy.deepcopy(self.dist_layout)
        return t_

    # bind the new methods to the tensor
    dtensor.detach = new_detach.__get__(dtensor)
    dtensor.clone = new_clone.__get__(dtensor)
    return dtensor


def _construct_default_sharding_spec(
    tensor: torch.Tensor,
) -> ShardingSpec:
    """
    Construct the default sharding specification for the tensor.

    Args:
        tensor (`torch.Tensor`): the tensor to be sharded.

    Returns:
        A `ShardingSpec` object without any sharding specified.
    """
    return ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={})


def _apply_layout(tensor, layout):
    """
    Apply the layout to the local tensor during initializing process.
    """
    # layout converter requires a source and target layout
    # we construct the source layer for an unsharded tensor
    # and use self.dist_layer as the target layout for the sharded tensor
    source_spec = _construct_default_sharding_spec(tensor)
    source_layout = Layout(device_mesh=layout.device_mesh, sharding_spec=source_spec, global_shape=tensor.shape)
    sharded_tensor = layout_converter.apply(tensor=tensor, source_layout=source_layout, target_layout=layout)
    return sharded_tensor


def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh, sharding_spec: ShardingSpec) -> torch.Tensor:
    """
    Convert the given tensor to a distributed tensor.

    Args:
        tensor (torch.Tensor): The tensor to be converted.
        device_mesh (DeviceMesh): The device mesh for abstraction of the compute devices.
        sharding_spec (ShardingSpec): The sharding specification which describes how the tensor will be sharded.

    Returns:
        torch.Tensor: The distributed tensor.
    """
    assert not is_distributed_tensor(tensor), "The input tensor is already a distributed tensor."
    dist_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec, global_shape=tensor.shape)

    # shard tensor
    sharded_tensor = _apply_layout(tensor, dist_layout)

    # hack some tensor methods
    _hijack_detach_and_clone(sharded_tensor)

    return sharded_tensor


def init_as_dtensor(
    tensor: torch.Tensor, device_mesh: DeviceMesh, sharding_spec: ShardingSpec, global_shape: torch.Size
) -> torch.Tensor:
    assert not is_distributed_tensor(tensor), "The input tensor is already a distributed tensor."
    dist_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec, global_shape=global_shape)

    # shard tensor
    tensor.dist_layout = dist_layout

    # hack some tensor methods
    _hijack_detach_and_clone(tensor)

    return tensor


def redistribute(dtensor: torch.Tensor, device_mesh: DeviceMesh, sharding_spec: ShardingSpec) -> None:
    """
    Convert the layout of the tensor from source_spec to target_spec.
    This will update the `local_tensor` and `dist_layout` in place.

    Args:
        dtensor (torch.Tensor): the distributed tensor to be converted.
        device_mesh (DeviceMesh): the device mesh for abstraction of the compute devices.
        target_layout (Layout): the target layout specification.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    global_shape = get_global_shape(dtensor)
    target_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec, global_shape=global_shape)
    resharded_tensor = layout_converter.apply(
        tensor=dtensor, source_layout=dtensor.dist_layout, target_layout=target_layout
    )
    return resharded_tensor


def to_global(dtensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a distributed tensor to the global tensor with the given layout.
    This function returns a native `torch.Tensor` object.

    Args:
        dtensor (torch.Tensor): the distributed tensor to be converted.

    Returns:
        torch.Tensor: the global tensor.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    layout_converter = LayoutConverter()

    global_sharding_spec = ShardingSpec(dtensor.dim(), {})
    device_mesh = get_device_mesh(dtensor)
    global_shape = get_global_shape(dtensor)
    global_layout = Layout(device_mesh=device_mesh, sharding_spec=global_sharding_spec, global_shape=global_shape)

    global_tensor = layout_converter.apply(dtensor, dtensor.dist_layout, global_layout)
    return global_tensor


def shard_rowwise(
    tensor: torch.Tensor,
    group_or_device_mesh: Union[ProcessGroup, DeviceMesh] = None,
) -> torch.Tensor:
    """
    Shard the first dim of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be sharded.
        group_or_device_mesh (Union[ProcessGroup, DeviceMesh], optional): The group or device mesh to shard the tensor.
            If None, the tensor will be sharded with respect to the global process group.
            Defaults to None.
        inplace (bool, optional): Whether to shard the tensor in-place. Defaults to False.

    Returns:
        torch.Tensor: The sharded tensor.
    """
    # if the group_or_device_mesh is None, we shard the tensor with respect to the global process group
    if group_or_device_mesh is None:
        group_or_device_mesh = dist.GroupMember.WORLD

    if isinstance(group_or_device_mesh, ProcessGroup):
        device_mesh = DeviceMesh.from_process_group(group_or_device_mesh)
    else:
        assert len(group_or_device_mesh.shape) == 1, "Only 1D DeviceMesh is accepted for row-wise sharding."
        device_mesh = group_or_device_mesh

    sharding_spec = ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={0: [0]})

    return distribute_tensor(tensor, device_mesh, sharding_spec)


def shard_colwise(tensor: torch.Tensor, group_or_device_mesh: Union[ProcessGroup, DeviceMesh] = None) -> torch.Tensor:
    """
    Shard the first dim of the given tensor.

    Args:
        tensor (torch.Tensor): The tensor to be sharded.
        group_or_device_mesh (Union[ProcessGroup, DeviceMesh], optional): The group or device mesh to shard the tensor.
            If None, the tensor will be sharded with respect to the global process group.
            Defaults to None.
        inplace (bool, optional): Whether to shard the tensor in-place. Defaults to False.

    Returns:
        torch.Tensor: The sharded tensor.
    """
    # if the group_or_device_mesh is None, we shard the tensor with respect to the global process group
    if group_or_device_mesh is None:
        group_or_device_mesh = dist.GroupMember.WORLD

    if isinstance(group_or_device_mesh, ProcessGroup):
        device_mesh = DeviceMesh.from_process_group(group_or_device_mesh)
    else:
        assert len(group_or_device_mesh.shape) == 1, "Only 1D DeviceMesh is accepted for row-wise sharding."
        device_mesh = group_or_device_mesh
    sharding_spec = ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={-1: [0]})

    return distribute_tensor(tensor, device_mesh, sharding_spec)


def sharded_tensor_to_param(dtensor: torch.Tensor, requires_grad: bool = True):
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    param = torch.nn.Parameter(dtensor, requires_grad=requires_grad)

    # make it distributed as well
    param.dist_layout = dtensor.dist_layout
    _hijack_detach_and_clone(param)

    return param


def sharded_tensor_to_existing_param(dtensor: torch.Tensor, param: torch.nn.Parameter) -> None:
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    param.data = dtensor
    # make it distributed as well
    param.dist_layout = dtensor.dist_layout
    _hijack_detach_and_clone(param)


def compute_global_numel(dtensor: torch.Tensor) -> int:
    """
    Compute the global number of elements in the distributed tensor.

    Args:
        dtensor (torch.Tensor): The distributed tensor.

    Returns:
        int: The global number of elements in the distributed tensor.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    numel = reduce(operator.mul, dtensor.dist_layout.global_shape)
    return numel


def get_layout(dtensor: torch.Tensor) -> Layout:
    """
    Get the layout of the distributed tensor.

    Args:
        dtensor (torch.Tensor): The distributed tensor.

    Returns:
        Layout: The layout of the distributed tensor.

    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    return dtensor.dist_layout


def get_global_shape(dtensor: torch.Tensor) -> torch.Size:
    """
    Get the global shape of the distributed tensor.

    Args:
        dtensor (torch.Tensor): The distributed tensor.

    Returns:
        torch.Size: The global shape of the distributed tensor.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    return dtensor.dist_layout.global_shape


def get_device_mesh(dtensor: torch.Tensor) -> DeviceMesh:
    """
    Get the device mesh of the distributed tensor.

    Args:
        dtensor (torch.Tensor): The distributed tensor.

    Returns:
        DeviceMesh: The device mesh of the distributed tensor.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    return dtensor.dist_layout.device_mesh


def get_sharding_spec(dtensor: torch.Tensor) -> ShardingSpec:
    """
    Get the sharding spec of the distributed tensor.

    Args:
        dtensor (torch.Tensor): The distributed tensor.

    Returns:
        ShardingSpec: The sharding spec of the distributed tensor.
    """
    assert is_distributed_tensor(dtensor), "The input tensor is not a distributed tensor."
    return dtensor.dist_layout.sharding_spec


# ======================================================
# Some sharding does not obey the SPMD style
# e.g. Fused QKV layer in GPT2
# we support customize sharding with the following APIs
# ======================================================
def is_customized_distributed_tensor(tensor: torch.Tensor):
    """
    Check whether the given tensor is a customized distributed tensor.

    Args:
        tensor (torch.Tensor): The tensor to be checked.

    Returns:
        bool: Whether the given tensor is a customized distributed tensor.
    """
    return hasattr(tensor, "shard_fn") and hasattr(tensor, "gather_fn")


def _hijack_detach_and_clone_for_customized_distributed_tensor(dtensor: torch.Tensor) -> torch.Tensor:
    """
    Hijack the detach and clone methods of the tensor to make sure the dist_layout is copied.

    Args:
        tensor (torch.Tensor): The tensor to be hijacked.

    Returns:
        torch.Tensor: The hijacked tensor.
    """
    dtensor._old_detach = dtensor.detach
    dtensor._old_clone = dtensor.clone

    def new_detach(self):
        t_ = self._old_detach()
        t_.shard_fn = self.shard_fn
        t_.gather_fn = self.gather_fn
        return t_

    def new_clone(self, *args, **kwargs):
        t_ = self._old_clone(*args, **kwargs)
        t_.shard_fn = self.shard_fn
        t_.gather_fn = self.gather_fn
        return t_

    # bind the new methods to the tensor
    dtensor.detach = new_detach.__get__(dtensor)
    dtensor.clone = new_clone.__get__(dtensor)
    return dtensor


def distribute_tensor_with_customization(tensor: torch.Tensor, shard_fn, gather_fn: callable):
    """
    Distribute the given tensor with the given shard_fn and gather_fn.

    Example:

    ```python
    # define shard and gather functions
    def shard_fn(tensor):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return tensor.chunk(world_size, dim=0)[rank]

    def gather_fn(tensor):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        shard_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(shard_list, tensor)
        return torch.cat(shard_list, dim=0)

    # create a distributed tensor
    tensor = torch.rand(4, 4)
    dtensor = distribute_tensor_with_customization(tensor, shard_fn, gather_fn)
    ```

    Args:
        tensor (torch.Tensor): The tensor to be distributed.
        shard_fn (callable): The function to shard the tensor.
        gather_fn (callable): The function to gather the tensor.

    Returns:
        torch.Tensor: The distributed tensor.
    """
    assert callable(shard_fn), "The shard_fn must be callable."
    assert callable(gather_fn), "The gather_fn must be callable."
    assert not is_distributed_tensor(tensor), "The input tensor is already a distributed tensor."

    sharded_tensor = shard_fn(tensor)

    # set the shard_fn and gather_fn as attributes of the distributed tensor
    sharded_tensor.shard_fn = shard_fn
    sharded_tensor.gather_fn = gather_fn

    # set the shard_fn and gather_fn as attributes of the distributed tensor
    _hijack_detach_and_clone_for_customized_distributed_tensor(sharded_tensor)

    return sharded_tensor


def init_tensor_as_customization_distributed(tensor: torch.Tensor, shard_fn, gather_fn: callable):
    """
    Distribute the given tensor with the given shard_fn and gather_fn.

    Example:

    ```python
    # define shard and gather functions
    def shard_fn(tensor):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return tensor.chunk(world_size, dim=0)[rank]

    def gather_fn(tensor):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        shard_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(shard_list, tensor)
        return torch.cat(shard_list, dim=0)

    # create a distributed tensor
    tensor = torch.rand(4, 4)
    dtensor = init_tensor_as_customization_distributed(tensor, shard_fn, gather_fn)
    ```

    Args:
        tensor (torch.Tensor): The tensor to be distributed.
        shard_fn (callable): The function to shard the tensor.
        gather_fn (callable): The function to gather the tensor.

    Returns:
        torch.Tensor: The distributed tensor.
    """
    assert callable(shard_fn), "The shard_fn must be callable."
    assert callable(gather_fn), "The gather_fn must be callable."
    assert not is_distributed_tensor(tensor), "The input tensor is already a distributed tensor."

    # set the shard_fn and gather_fn as attributes of the distributed tensor
    tensor.shard_fn = shard_fn
    tensor.gather_fn = gather_fn

    # set the shard_fn and gather_fn as attributes of the distributed tensor
    _hijack_detach_and_clone_for_customized_distributed_tensor(tensor)

    return tensor


def to_global_for_customized_distributed_tensor(dtensor: torch.Tensor) -> torch.Tensor:
    """
    Gather the given tensor to the global tensor.

    Args:
        dtensor (torch.Tensor): The distributed tensor.

    Returns:
        torch.Tensor: The global tensor.
    """
    assert is_customized_distributed_tensor(dtensor), "The input tensor is not a customized distributed tensor."
    return dtensor.gather_fn(dtensor)


def customized_distributed_tensor_to_param(dtensor: torch.Tensor, requires_grad: bool = True):
    """
    Convert the given customized distributed tensor to a parameter.
    """
    assert is_customized_distributed_tensor(dtensor), "The input tensor is not a customized distributed tensor."

    param = torch.nn.Parameter(dtensor, requires_grad=requires_grad)

    # make it distributed as well
    param.shard_fn = dtensor.shard_fn
    param.gather_fn = dtensor.gather_fn
    _hijack_detach_and_clone_for_customized_distributed_tensor(param)
    return param


def customized_distributed_tensor_to_existing_param(dtensor: torch.Tensor, param: torch.nn.Parameter):
    """
    Convert the given customized distributed tensor to an existing parameter.
    """
    assert is_customized_distributed_tensor(dtensor), "The input tensor is not a customized distributed tensor."

    param.data = dtensor.data
    param.shard_fn = dtensor.shard_fn
    param.gather_fn = dtensor.gather_fn
    _hijack_detach_and_clone_for_customized_distributed_tensor(param)

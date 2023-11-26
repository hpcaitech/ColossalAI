from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn

from colossalai.tensor.colo_tensor import ColoTensor


def all_gather_simulator(target_pair):
    """
    Simulating all-gather operation, analyze the communication cost
    and simulate the influence of the DimSpec.

    We don't allow uncontiguous layout, such as all-gather(S012)->S02 is NOT allowed.
    Therefore, all gather operation just remove the last element in shard list,
    e.g.:
        all-gather(S01) -> S0

    Argument:
        target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
        and the second element describes which logical axis will be sharded in that dimension.
    """
    _, shard_list = target_pair
    new_shard_list = shard_list[:-1]

    return new_shard_list


def all_to_all_simulator(f_target_pair, b_target_pair):
    """
    Simulating all-to-all operation, analyze the communication cost
    and simulate the influence of the DimSpec.

    We BANNED all representations which shard_list in decreasing order,
    such as S10, so all-to-all(S0, S1) -> RS01 is NOT allowed.
    Therefore, if the behind shard_list is not None, we just extend it to the front shard_list.
    Argument:
        target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
        and the second element describes which logical axis will be sharded in that dimension.
    e.g.:
        all-to-all(S0, S1) -> [S01, R]
        all-to-all(S0, R) -> [R, S0]
    Otherwise, we extend the front shard_list to behind.
    e.g.:
        all-to-all(R, S1) -> [S1, R]

    Argument:
        target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
        and the second element describes which logical axis will be sharded in that dimension.
    """
    _, f_shard_list = f_target_pair
    _, b_shard_list = b_target_pair
    if not len(b_shard_list):
        b_shard_list.extend(f_shard_list)
        f_shard_list = []
    else:
        f_shard_list.extend(b_shard_list)
        b_shard_list = []

    return f_shard_list, b_shard_list


def shard_simulator(target_pair, legal_sharding_dims):
    """
    Simulating shard operation, analyze the communication cost(always ZERO)
    and simulate the influence of the DimSpec.

    We don't allow uncontiguous layout, such as shard(S0)->S02 is NOT allowed.
    In addition, We BANNED all representations which shard_list in decreasing order,
    such as S10, so shard(S0) -> S10 is NOT allowed.
    Therefore, for the R dimension, we could just append any legal sharding dim on it.
    e.g.:
        shard(R) -> S0
    For the S dimension, we need to make sure the shard_list after sharding still keep rising order.
    e.g:
        shard(S0) -> S01

    Argument:
        target_pair(Tuple[int, List[int]]): The first element is the dimension of tensor to be sharded,
        and the second element describes which logical axis will be sharded in that dimension.
    """
    _, shard_list = target_pair
    shard_list_list = []
    for dim in legal_sharding_dims:
        if len(shard_list) != 0 and dim <= shard_list[-1]:
            continue
        new_shard_list = shard_list + [dim]
        shard_list_list.append(new_shard_list)

    return shard_list_list


def mix_gather_simulator(f_target_pair, b_target_pair):
    """
    Assume index of f and b target pairs are 'f' and 'b'
    S0S1 => Input: (f, [0]), (b, [1]) Output: [b, f], (1, 0)
    S1S0 => Input: (f, [1]), (b, [0]) Output: [b, f], (0, 1)
    S01R => Input: (f, [0, 1]), (b, []) Output: [f], (1, 1)
    RS01 => Input: (f, []), (b, [0, 1]) Output: [b], (1, 1)
    S10R => Input: (f, [0, 1]), (b, []) Output: [f], (0, 0)
    RS10 => Input: (f, []), (b, [0, 1]) Output: [b], (0, 0)
    """
    if f_target_pair[1] and b_target_pair[1]:
        leading_dim = b_target_pair[1] > f_target_pair[1]
        return [b_target_pair[0], f_target_pair[0]], [int(leading_dim), int(leading_dim ^ 1)]
    if f_target_pair[1]:
        leading_dim = f_target_pair[1][0] < f_target_pair[1][1]
        return [
            f_target_pair[0],
        ], [int(leading_dim), int(leading_dim)]
    if b_target_pair[1]:
        leading_dim = b_target_pair[1][0] < b_target_pair[1][1]
        return [
            b_target_pair[0],
        ], [int(leading_dim), int(leading_dim)]


# The function is credited to PyTorch Team
def named_params_with_colotensor(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ColoTensor]]]:
    r"""Returns an iterator over module parameters (together with the
    ColoTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to a
    :class:torchshard._shard.sharded_optim.ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (string, Union[Tensor, ColoTensor]): Tuple containing
            the name and parameter (or ColoTensor parameter)

    Example:

        >>> model = torch.nn.Linear(*linear_size)
        >>> delattr(model.weight)
        >>> setattr(model.weight, ColoTensor(...))
        >>> for name, param in named_params_with_colotensor(model):
        >>>    if name in ['weight']:
        >>>        print(param.size())

    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    memo = set()
    for mod_prefix, mod in modules:
        # find all sharded tensor params
        for name, val in vars(mod).items():
            if isinstance(val, ColoTensor) and val not in memo:
                memo.add(val)
                name = mod_prefix + ("." if mod_prefix else "") + name
                yield name, val

    # find all nn.Parameters
    for name, val in module.named_parameters():
        yield name, val


def _convert_tensor(tensor: torch.Tensor) -> ColoTensor:
    return ColoTensor(tensor)


def convert_parameter(module: torch.nn.Module, param_name: str):
    # Perform some validation first.
    if not hasattr(module, param_name):
        raise ValueError(f"module: {module} does not have parameter with name: {param_name}")

    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}"
        )

    if not tensor.is_contiguous():
        raise ValueError(f"param: {param_name} is not a contiguous Tensor")

    st = _convert_tensor(tensor)

    # Replace param with ColoTensor.

    # Need to delete the attribute first since param_name might be
    # torch.nn.Parameter and can't be replaced with ColoTensor which is
    # not torch.nn.Parameter.
    delattr(module, param_name)

    # Now we can set the attribute appropriately.
    setattr(module, param_name, st)


def convert_dim_partition_dict(dim_size: int, dim_partition_dict: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    This method is used to convert the negative dim value to positive.
    """
    dims_to_convert = []
    for dim, mesh_list in dim_partition_dict.items():
        if dim < 0:
            dims_to_convert.append(dim)
    for dim in dims_to_convert:
        dim_partition_dict.pop(dim)
        dim_partition_dict[dim_size + dim] = mesh_list
    return dim_partition_dict


def merge_same_dim_mesh_list(dim_size: int, dim_partition_dict: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    This method is used to merge the different key value which points to same physical position.

    For example:
        dim_partition_dict: {1 :[0], -1: [1]} or {1: [0], 1: [1]} for a 2d tensor, the dim 1 and -1 point same physical position.
        In this method, above dim_partition_dict will be converted to {1: [0, 1]}
    """
    converted_dim_partition_dict = {}
    for dim, mesh_list in dim_partition_dict.items():
        if dim < 0:
            dim = dim_size + dim
        if dim not in converted_dim_partition_dict:
            converted_dim_partition_dict[dim] = mesh_list
        else:
            converted_dim_partition_dict[dim].extend(mesh_list)

    return converted_dim_partition_dict

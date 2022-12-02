from enum import Enum
from typing import Dict, List, Tuple

import torch


class PreviousStatus(Enum):
    """
    This class shows the status of previous comparision.
    """
    RESET = 0
    # ORIGIN means the dimension size of original tensor is larger in the previous comparision.
    ORIGIN = 1
    # TGT means the dimension size of target tensor is larger in the previous comparision.
    TGT = 2


def detect_reshape_mapping(origin_shape: torch.Size, tgt_shape: torch.Size) -> Dict[Tuple[int], Tuple[int]]:
    """
    This method is used to detect the reshape mapping between original tensor and target tensor.

    Returns:
        reshape_mapping_dict: The dictionary shows how a tuple of origin dims(keys) mapping to the related
        target dims(values) during reshaping operation.
    Examples:
        import torch
        origin_shape = torch.Size([4, 4, 4])
        tgt_shape = torch.Size([2, 8, 2, 2])
        reshape_mapping_dict = detect_reshape_mapping(origin_shape, tgt_shape)
        print(reshape_mapping_dict)
    Output:
        {(2,): (3, 2), (1, 0): (1,), (0,): (0, 1)}
    """

    # reverse the shape object
    origin_shape = list(origin_shape)
    tgt_shape = list(tgt_shape)
    origin_shape.reverse()
    tgt_shape.reverse()

    # initialize arguments
    reshape_mapping_dict = {}
    origin_len = len(origin_shape)
    tgt_len = len(tgt_shape)
    origin_index = 0
    tgt_index = 0
    original_dimension_size = origin_shape[origin_index]
    tgt_dimension_size = tgt_shape[tgt_index]
    tgt_dims = [tgt_len - tgt_index - 1]
    origin_dims = [origin_len - origin_index - 1]
    previous_label = PreviousStatus.RESET

    while origin_index != len(origin_shape) or tgt_index != len(tgt_shape):
        if original_dimension_size == tgt_dimension_size:
            reshape_mapping_dict[tuple(origin_dims)] = tuple(tgt_dims)
            # if the origin_dims has no element, it means the original tensor has been fully matched.
            # Therefore, we do not have to increase the origin_index for that case.
            if len(origin_dims) > 0:
                origin_index += 1
            # if the tgt_dims has no element, it means the original tensor has been fully matched.
            # Therefore, we do not have to increase the tgt_index for that case.
            if len(tgt_dims) > 0:
                tgt_index += 1
            # the last step of loop should always end with condition
            # so we need to manually skip the preparation for next step
            # in the last step.
            if origin_index == len(origin_shape) and tgt_index == len(tgt_shape):
                continue

            # If origin_index equals to origin_len, we just need to set the original_dimension_size
            # to 1 to match the remaining '1's in the target tensor shape.
            if origin_index == len(origin_shape):
                original_dimension_size = 1
                origin_dims = []
            else:
                original_dimension_size = origin_shape[origin_index]
                origin_dims = [origin_len - origin_index - 1]

            # If tgt_index equals to tgt_len, we just need to set the tgt_dimension_size
            # to 1 to match the remaining '1's in the original tensor shape.
            if tgt_index == len(tgt_shape):
                tgt_dimension_size = 1
                tgt_dims = []
            else:
                tgt_dimension_size = tgt_shape[tgt_index]
                tgt_dims = [tgt_len - tgt_index - 1]

            previous_label = PreviousStatus.RESET

        elif original_dimension_size > tgt_dimension_size:
            tgt_index += 1

            if previous_label == PreviousStatus.TGT:
                # if the target dimension size is larger in the previous comparision, which means
                # the origin dimension size has already accumulated larger than target dimension size, so
                # we need to offload the origin dims and tgt dims into the reshape_mapping_dict.
                reshape_mapping_dict[tuple(origin_dims)] = tuple(tgt_dims)
                original_dimension_size = original_dimension_size // tgt_dimension_size
                origin_dims = [origin_len - origin_index - 1]
                tgt_dimension_size = tgt_shape[tgt_index]
                tgt_dims = [tgt_len - tgt_index - 1, tgt_len - tgt_index]
                # reset the previous_label after offloading the origin dims and tgt dims
                previous_label = PreviousStatus.RESET
            else:
                # accumulate the tgt_dimension_size until tgt_dimension_size larger than original_dimension_size
                tgt_dimension_size *= tgt_shape[tgt_index]
                tgt_dims.append(tgt_len - tgt_index - 1)
                previous_label = PreviousStatus.ORIGIN

        else:
            origin_index += 1

            if previous_label == PreviousStatus.ORIGIN:
                # if the origin element is larger in the previous comparision, which means
                # the target element has already accumulated larger than origin element, so
                # we need to offload the origin dims and tgt dims into the reshape_mapping_dict.
                reshape_mapping_dict[tuple(origin_dims)] = tuple(tgt_dims)
                tgt_dimension_size = tgt_dimension_size // original_dimension_size
                tgt_dims = [tgt_len - tgt_index - 1]
                original_dimension_size = origin_shape[origin_index]
                origin_dims = [origin_len - origin_index - 1, origin_len - origin_index]
                # reset the previous_label after offloading the origin dims and tgt dims
                previous_label = PreviousStatus.RESET
            else:
                # accumulate the original_dimension_size until original_dimension_size larger than tgt_dimension_size
                original_dimension_size *= origin_shape[origin_index]
                origin_dims.append(origin_len - origin_index - 1)
                previous_label = PreviousStatus.TGT

    return reshape_mapping_dict


def check_keep_sharding_status(input_dim_partition_dict: Dict[int, List[int]],
                               reshape_mapping_dict: Dict[Tuple[int], Tuple[int]]) -> bool:
    """
    This method is used to check whether the reshape operation could implement without converting
    the input to fully replicated status.

    Rule:
        For a sharded dimension of input tensor, if it is not the minimum element of the input tuple,
        the function will return false.
        To illustrate this issue, there are two cases to analyse:
        1. no sharded dims in the input tuple: we could do the reshape operation safely just as the normal
        operation without distributed tensor.
        2. sharded dims in the input tuple: the sharded dim must be the minimum element, then during shape
        consistency process, torch.cat will be implemented on the sharded dim, and everything after the sharded
        dim get recovered.

    Examples:
        # the second dimension of the input has been sharded.
        input_dim_partition_dict = {1: [1]}
        origin_shape = torch.Size([8, 4, 2])
        tgt_shape = torch.Size([2, 4, 8])
        reshape_mapping_dict = detect_reshape_mapping(origin_shape, tgt_shape)
        # {(2, 1): (2,), (0,): (1, 0)}
        # the sharded dim of input is 1, which is the minimum element of the tuple (2, 1),
        # so we do not have to convert the input to fully replicated status.
        print(check_keep_sharding_status(input_dim_partition_dict, reshape_mapping_dict))

    Output:
        True
    """
    sharded_dims = list(input_dim_partition_dict.keys())
    for input_dims in reshape_mapping_dict.keys():
        # if input_dims has no element, we could just skip this iteration.
        if len(input_dims) == 0:
            continue
        min_element = min(input_dims)
        for dim in input_dims:
            if dim in sharded_dims and dim is not min_element:
                return False
    return True


def infer_output_dim_partition_dict(input_dim_partition_dict: Dict[int, List[int]],
                                    reshape_mapping_dict: Dict[Tuple[int], Tuple[int]]) -> Dict[Tuple[int], Tuple[int]]:
    """
    This method is used to infer the output dim partition dict for a reshape operation,
    given the input dim partition dict and reshape mapping dict.
    """
    assert check_keep_sharding_status(input_dim_partition_dict, reshape_mapping_dict), \
        'we only infer output dim partition dict for the reshape operation could keep sharding spec.'
    sharded_dims = list(input_dim_partition_dict.keys())
    output_dim_partition_dict = {}
    for input_dims, output_dims in reshape_mapping_dict.items():
        for dim in input_dims:
            if dim in sharded_dims:
                output_dim_partition_dict[min(output_dims)] = input_dim_partition_dict[dim]
                # we could break because input dims cannot contain two sharded dims, otherwise
                # the keep sharding status check will fail.
                break
    return output_dim_partition_dict

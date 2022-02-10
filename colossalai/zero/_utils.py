from numpy import partition
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from colossalai.utils import get_current_device


def to_cpu(input_):
    if isinstance(input_, (list, tuple)):
        for tensor in input_:
            tensor.data = tensor.data.cpu()
    elif torch.is_tensor(input_):
        input_.data = input_.data.cpu()
    else:
        raise TypeError(
            f"Expected argument 'input_' to be torch.Tensor, list or tuple, but got {type(input_)} "
        )


def flatten(input_):
    return _flatten_dense_tensors(input_)


def unflatten(flat, tensors):
    return _unflatten_dense_tensors(flat, tensors)


def count_numel(tensor_list):
    res = 0
    for tensor in tensor_list:
        res += tensor.numel()
    return res


def calculate_padding(numel, unit_size):
    remainder = numel % unit_size
    return unit_size - remainder if remainder else remainder


def shuffle_by_round_robin(tensor_list, num_partitions):
    partitions = dict()

    for tensor_idx, tensor in enumerate(tensor_list):
        partition_to_go = tensor_idx % num_partitions
        if partition_to_go not in partitions:
            partitions[partition_to_go] = []
        partitions[partition_to_go].append(dict(tensor=tensor,
                                                index=tensor_idx))

    partitions_count = len(partitions)
    new_tensor_list = []
    tensor_index_mapping = dict()

    for partition_id in range(partitions_count):
        partition_tensors = partitions[partition_id]
        for item in partition_tensors:
            tensor_index_mapping[item['index']] = len(new_tensor_list)
            new_tensor_list.append(item['tensor'])

    return new_tensor_list, tensor_index_mapping


# create a flat tensor aligned at the alignment boundary
def flatten_dense_tensors_with_padding(self, tensor_list, unit_size):
    num_elements = count_numel(tensor_list)
    padding = calculate_padding(num_elements, unit_size=unit_size)

    if padding > 0:
        pad_tensor = torch.zeros(padding,
                                 device=tensor_list[0].device,
                                 dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list

    return flatten(padded_tensor_list)


def is_nccl_aligned(tensor):
    return tensor.data_ptr() % 4 == 0

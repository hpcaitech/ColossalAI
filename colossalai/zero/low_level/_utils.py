import math
from typing import Optional

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


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
        partitions[partition_to_go].append(dict(tensor=tensor, index=tensor_idx))

    partitions_count = len(partitions)
    new_tensor_list = []
    tensor_index_mapping = dict()

    for partition_id in range(partitions_count):
        partition_tensors = partitions[partition_id]
        for item in partition_tensors:
            tensor_index_mapping[item["index"]] = len(new_tensor_list)
            new_tensor_list.append(item["tensor"])

    return new_tensor_list, tensor_index_mapping


# create a flat tensor aligned at the alignment boundary
def flatten_dense_tensors_with_padding(tensor_list, unit_size):
    num_elements = count_numel(tensor_list)
    padding = calculate_padding(num_elements, unit_size=unit_size)

    if padding > 0:
        pad_tensor = torch.zeros(padding, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list

    return flatten(padded_tensor_list)


def is_nccl_aligned(tensor):
    return tensor.data_ptr() % 4 == 0


def get_grad_accumulate_object(tensor):
    """
    Return the AccumulateGrad of the input tensor
    """

    # grad_fn reference:
    # https://discuss.pytorch.org/t/in-the-grad-fn-i-find-a-next-functions-but-i-dont-understand-the-meaning-of-the-attribute/24463
    # expand_as reference: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    #
    # `next_functions` will return the backward graph where
    # the first element is the AccumulateGrad of the leaf nodes.
    # we want to get the AccumulateGrad of the input tensor instead of the leaf
    # node in the whole computation graph.
    # Therefore, we call expand_as to create a dummy graph
    # where tensor_tmp and tensor indeed point to the same object.
    # You can check this by print(tensor.data_ptr() == tensor_tmp.data_ptr())
    tensor_tmp = tensor.expand_as(tensor)
    grad_acc_obj = tensor_tmp.grad_fn.next_functions[0][0]
    return grad_acc_obj


def split_by_dtype(tensor_list):
    """
    Splits a list of PyTorch tensors into sublists based on their data type.

    :param tensor_list: A list of PyTorch tensors.
    :type tensor_list: list[torch.Tensor]
    :return: A list of sublists, where each sublist contains tensors of a specific data type.
    :rtype: list[list[torch.Tensor]]
    """
    dtypes = ["torch.cuda.HalfTensor", "torch.cuda.FloatTensor", "torch.cuda.DoubleTensor", "torch.cuda.BFloat16Tensor"]
    buckets = []
    for _, dtype in enumerate(dtypes):
        bucket = [t for t in tensor_list if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def reduce_tensor_dp_group(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    dst_local_rank: Optional[int] = None,
    dst_global_rank: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Reduce the tensor in the data parallel process group

    :param tensor: A tensor object to reduce/all-reduce
    :param dtype: The data type used in communication
    :param dst_rank: The source rank for reduce. If dst_rank is None,
    :param parallel_mode: Communication parallel mode
    all-reduce will be used instead of reduce. Default is None.

    :type tensor: torch.Tensor
    :type dtype: torch.dtype, optional
    :type dst_rank: int, optional
    :type pg: ProcessGroup, optional
    """
    # use the original dtype
    if dtype is None:
        dtype = tensor.dtype

    # cast the data to specified dtype for reduce/all-reduce
    if tensor.dtype != dtype:
        tensor_to_reduce = tensor.to(dtype)
    else:
        tensor_to_reduce = tensor

    world_size = dist.get_world_size(group=group)
    tensor_to_reduce.div_(world_size)

    # if rank is None, all reduce will be used
    # else, reduce is used
    use_all_reduce = dst_local_rank is None

    if use_all_reduce:
        dist.all_reduce(tensor_to_reduce, group=group)
    else:
        dist.reduce(tensor=tensor_to_reduce, dst=dst_global_rank, group=group)

    # recover the original dtype
    if tensor.dtype != dtype and tensor is not tensor_to_reduce:
        local_rank = dist.get_rank(group=group)
        if use_all_reduce or dst_local_rank == local_rank:
            tensor.copy_(tensor_to_reduce)

    return tensor


def has_inf_or_nan(tensor):
    try:
        # if tensor is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as tensor
        # (which is true for some recent version of pytorch).
        tensor_sum = float(tensor.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # tensor_sum = float(tensor.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if tensor_sum == float("inf") or tensor_sum == -float("inf") or tensor_sum != tensor_sum:
            return True
        return False


def release_param_grad(tensor_list):
    for tensor in tensor_list:
        tensor.grad = None


def calculate_global_norm_from_list(norm_list):
    """Compute total from a list of norms"""
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    return math.sqrt(total_norm)


def sync_tensor(flat_tensor, tensor_list):
    """
    Synchronize the flattened tensor and unflattened tensor list. When
    a list of tensor are flattened with `torch._utils._unflatten_dense_tensors`,
    a new tensor is created. Thus, the flat tensor and original tensor list do not
    share the same memory space. This function will update the tensor list so that
    they point to the same value.

    :param flat_tensor: A flat tensor obtained by calling `torch._utils._unflatten_dense_tensors` on a tensor list
    :param tensor_list: A list of tensors corresponding to the flattened tensor
    :type flat_tensor: torch.Tensor
    :type tensor_list: List[torch.Tensor]
    """
    updated_params = unflatten(flat_tensor, tensor_list)

    # update the tensor data
    for p, q in zip(tensor_list, updated_params):
        p.data = q.data

import torch

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.core import global_context as gpc

_MAX_DATA_DIM = 5


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if not gpc.is_initialized(ParallelMode.TENSOR) or gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, "you should increase MAX_DATA_DIM"
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.cuda.LongTensor(sizes)
    torch.distributed.broadcast(
        sizes_cuda, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], group=gpc.get_group(ParallelMode.TENSOR)
    )

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data dictionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    # Pack on rank zero.
    if not gpc.is_initialized(ParallelMode.TENSOR) or gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        # Check that all keys have the same data type.
        # Flatten the data associated with the keys
        flatten_data = torch.cat([data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
    else:
        flatten_data = torch.empty(total_numel, device=torch.cuda.current_device(), dtype=datatype)

    # Broadcast
    torch.distributed.broadcast(
        flatten_data, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], group=gpc.get_group(ParallelMode.TENSOR)
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ["text", "types", "labels", "is_random", "loss_mask", "padding_mask"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b["text"].long()
    types = data_b["types"].long()
    sentence_order = data_b["is_random"].long()
    loss_mask = data_b["loss_mask"].float()
    lm_labels = data_b["labels"].long()
    padding_mask = data_b["padding_mask"].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def get_batch_for_sequence_parallel(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ["text", "types", "labels", "is_random", "loss_mask", "padding_mask"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    # unpack
    data_b = broadcast_data(keys, data, datatype)

    # # get tensor parallel local rank
    global_rank = torch.distributed.get_rank()
    local_world_size = 1 if not gpc.is_initialized(ParallelMode.TENSOR) else gpc.get_world_size(ParallelMode.TENSOR)
    local_rank = global_rank % local_world_size
    seq_length = data_b["text"].size(1)
    sub_seq_length = seq_length // local_world_size
    sub_seq_start = local_rank * sub_seq_length
    sub_seq_end = (local_rank + 1) * sub_seq_length
    #
    # # Unpack.
    tokens = data_b["text"][:, sub_seq_start:sub_seq_end].long()
    types = data_b["types"][:, sub_seq_start:sub_seq_end].long()
    sentence_order = data_b["is_random"].long()
    loss_mask = data_b["loss_mask"][:, sub_seq_start:sub_seq_end].float()
    lm_labels = data_b["labels"][:, sub_seq_start:sub_seq_end].long()
    padding_mask = data_b["padding_mask"].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


class SequenceParallelDataIterator:
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        return self.data_iter

    def __next__(self):
        return get_batch_for_sequence_parallel(self.data_iter)

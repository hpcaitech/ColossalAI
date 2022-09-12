import torch.nn.functional as F
from typing import Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ComputePattern, ColoTensorSpec, ComputePattern, ComputeSpec, ColoTensor, ShardSpec, \
    ReplicaSpec
from ._utils import GeneralTensor, convert_to_colo_tensor, reduce_input


def colo_embedding_1Dcol(input_tensor: ColoTensor,
                         weight: ColoTensor,
                         padding_idx: Optional[int] = None,
                         max_norm: Optional[float] = None,
                         norm_type: float = 2.0,
                         scale_grad_by_freq: bool = False,
                         sparse: bool = False) -> ColoTensor:
    # embedding_1Dcol split the weight(lookup table) to (num_embeddings, embedding_dim/P)
    # Gather splitted lookup table
    input_tensor = input_tensor.redistribute(ReplicaSpec())

    output_parallel = F.embedding(input_tensor,
                                  weight,
                                  padding_idx=padding_idx,
                                  max_norm=max_norm,
                                  norm_type=norm_type,
                                  scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse)
    output_spec = ColoTensorSpec(weight.get_process_group(), ShardSpec([-1], [weight.get_tp_world_size()]),
                                 ComputeSpec(ComputePattern.TP1D))
    output = ColoTensor.from_torch_tensor(output_parallel, spec=output_spec)

    compute_spec = weight.compute_spec

    if compute_spec.output_replicate:
        return output.to_replicate()
    else:
        return output


def colo_embedding_1Drow(input_tensor: ColoTensor,
                         weight: ColoTensor,
                         padding_idx: Optional[int] = None,
                         max_norm: Optional[float] = None,
                         norm_type: float = 2.0,
                         scale_grad_by_freq: bool = False,
                         sparse: bool = False) -> ColoTensor:
    # embedding_1Drow splits the weight(lookup table) to the shape, [num_embeddings/P, embedding_dim]
    # get the index of current segment and mask other segments with 0

    # get complete input tensor through all-gather
    input_tensor = input_tensor.redistribute(ReplicaSpec())

    # tensor_parallel_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    tensor_parallel_rank = weight.get_process_group().tp_local_rank()
    num_embeddings_per_partition = weight.size_local(0)
    vocab_start_index = tensor_parallel_rank * num_embeddings_per_partition
    vocab_end_index = vocab_start_index + num_embeddings_per_partition

    # build the mask.
    input_mask = (input_tensor < vocab_start_index) | (input_tensor >= vocab_end_index)
    # mask the input.
    # TODO(jzy) masked_input may be an activation managed by ColoTensor.
    masked_input = input_tensor - vocab_start_index
    masked_input[input_mask] = 0

    partial_output = F.embedding(masked_input,
                                 weight,
                                 padding_idx=padding_idx,
                                 max_norm=max_norm,
                                 norm_type=norm_type,
                                 scale_grad_by_freq=scale_grad_by_freq,
                                 sparse=sparse)

    # Mask the output embedding.
    partial_output[input_mask, :] = 0.
    # Reduce across all the model parallel GPUs.
    output = reduce_input(partial_output, weight.get_process_group())
    output = ColoTensor.from_torch_tensor(output, spec=ColoTensorSpec(weight.get_process_group(), ReplicaSpec()))
    return output


def colo_embedding_1d(mode: str,
                      input_tensor: ColoTensor,
                      weight: ColoTensor,
                      padding_idx: Optional[int] = None,
                      max_norm: Optional[float] = None,
                      norm_type: float = 2.0,
                      scale_grad_by_freq: bool = False,
                      sparse: bool = False) -> ColoTensor:
    assert mode in ('row', 'col')
    funcs = {'row': colo_embedding_1Drow, 'col': colo_embedding_1Dcol}
    return funcs[mode](input_tensor,
                       weight,
                       padding_idx=padding_idx,
                       max_norm=max_norm,
                       norm_type=norm_type,
                       scale_grad_by_freq=scale_grad_by_freq,
                       sparse=sparse)


@colo_op_impl(F.embedding)
def colo_embedding(input_tensor: GeneralTensor,
                   weight: GeneralTensor,
                   padding_idx: Optional[int] = None,
                   max_norm: Optional[float] = None,
                   norm_type: float = 2.0,
                   scale_grad_by_freq: bool = False,
                   sparse: bool = False):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method looks up an embedding table.
    """
    assert isinstance(weight, ColoTensor)
    input_tensor = convert_to_colo_tensor(input_tensor, weight.get_process_group())

    if not weight.has_compute_spec():    # No Model Parallel Applied
        assert weight.is_replicate(), 'Invalid weight spec for native embedding op'
        return ColoTensor.from_torch_tensor(tensor=F.embedding(input_tensor,
                                                               weight,
                                                               padding_idx=padding_idx,
                                                               max_norm=max_norm,
                                                               norm_type=norm_type,
                                                               scale_grad_by_freq=scale_grad_by_freq,
                                                               sparse=sparse),
                                            spec=ColoTensorSpec(weight.get_process_group()))
    elif weight.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.is_shard_1drow():
            mode = 'row'
        elif weight.is_shard_1dcol():
            mode = 'col'
        else:
            raise NotImplementedError
        return colo_embedding_1d(mode,
                                 input_tensor,
                                 weight,
                                 padding_idx=padding_idx,
                                 max_norm=max_norm,
                                 norm_type=norm_type,
                                 scale_grad_by_freq=scale_grad_by_freq,
                                 sparse=sparse)
    else:
        raise NotImplementedError

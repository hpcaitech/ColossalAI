import torch.nn.functional as F
from typing import Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import reduce_input
from colossalai.core import global_context as gpc
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ComputeSpec, ColoTensor, distspec
from colossalai.context import ParallelMode
from ._utils import GeneralTensor, convert_to_colo_tensor


def colo_embedding_1Dcol(input_tensor: ColoTensor,
                         weight: ColoTensor,
                         padding_idx: Optional[int] = None,
                         max_norm: Optional[float] = None,
                         norm_type: float = 2.0,
                         scale_grad_by_freq: bool = False,
                         sparse: bool = False) -> ColoTensor:
    # embedding_1Dcol split the weight(lookup table) to (num_embeddings, embedding_dim/P)
    # Gather splitted lookup table
    input_tensor = input_tensor.convert_to_dist_spec(distspec.replicate(weight.spec.get_process_group()))

    output_parallel = F.embedding(input_tensor,
                                  weight,
                                  padding_idx=padding_idx,
                                  max_norm=max_norm,
                                  norm_type=norm_type,
                                  scale_grad_by_freq=scale_grad_by_freq,
                                  sparse=sparse)
    output_spec = TensorSpec(
        distspec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group_size()]),
        ComputeSpec(ComputePattern.TP1D))
    output = ColoTensor.from_torch_tensor(output_parallel, spec=output_spec)
    return output.to_replicate()


def colo_embedding_1Drow(input_tensor: ColoTensor,
                         weight: ColoTensor,
                         padding_idx: Optional[int] = None,
                         max_norm: Optional[float] = None,
                         norm_type: float = 2.0,
                         scale_grad_by_freq: bool = False,
                         sparse: bool = False) -> ColoTensor:
    # embedding_1Drow split the weight(lookup table) to (num_embeddings/P, embedding_dim)
    # Find index in this shard and mask those not here
    # Reduce all
    input_tensor = input_tensor.convert_to_dist_spec(distspec.replicate(weight.spec.get_process_group()))

    tensor_parallel_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    num_embeddings_per_partition = weight.size(0)
    vocab_start_index = tensor_parallel_rank * num_embeddings_per_partition
    vocab_end_index = vocab_start_index + num_embeddings_per_partition

    # Build the mask.
    input_mask = (input_tensor < vocab_start_index) | \
        (input_tensor >= vocab_end_index)
    # Mask the input.
    # TODO(jzy) masked_input may be an activation managed by ColoTensor.
    masked_input = input_tensor.clone() - vocab_start_index
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
    output = reduce_input(partial_output, ParallelMode.PARALLEL_1D)
    output = ColoTensor.from_torch_tensor(output, spec=TensorSpec(distspec.replicate(weight.spec.get_process_group())))
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
    input_tensor, weight = tuple(map(convert_to_colo_tensor, (input_tensor, weight)))

    # Handle differen parallel actions.

    if not weight.has_spec():    # No Model Parallel Applied
        assert weight.spec.is_gathered(), 'Invalid weight spec for native embedding op'
        return ColoTensor.from_torch_tensor(
            F.embedding(input_tensor,
                        weight,
                        padding_idx=padding_idx,
                        max_norm=max_norm,
                        norm_type=norm_type,
                        scale_grad_by_freq=scale_grad_by_freq,
                        sparse=sparse))
    elif weight.spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.spec.is_1D_row():
            mode = 'row'
        elif weight.spec.is_1D_col():
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

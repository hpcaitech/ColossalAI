import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ComputePattern, ComputePattern, ComputeSpec, ColoTensor, distspec, ColoTensorSpec, \
    ShardSpec, ReplicaSpec
from ._utils import GeneralTensor, convert_to_colo_tensor


def colo_embedding_bag_1Dcol(input_tensor: ColoTensor,
                             weight: ColoTensor,
                             offsets: Optional[Tensor] = None,
                             max_norm: Optional[float] = None,
                             norm_type: float = 2,
                             scale_grad_by_freq: bool = False,
                             mode: str = "mean",
                             sparse: bool = False,
                             per_sample_weights: Optional[Tensor] = None,
                             include_last_offset: bool = False,
                             padding_idx: Optional[int] = None) -> ColoTensor:
    # embedding_bag_1Dcol split the weight(lookup table) to (num_embeddings, embedding_dim/P)
    # Gather splitted lookup table
    pg = weight.get_process_group()
    input_tensor = input_tensor.redistribute(ReplicaSpec())

    output_parallel = F.embedding_bag(input_tensor,
                                      weight,
                                      offsets=offsets,
                                      max_norm=max_norm,
                                      norm_type=norm_type,
                                      scale_grad_by_freq=scale_grad_by_freq,
                                      mode=mode,
                                      sparse=sparse,
                                      per_sample_weights=per_sample_weights,
                                      include_last_offset=include_last_offset,
                                      padding_idx=padding_idx)
    output_spec = ColoTensorSpec(pg, ShardSpec([-1], [weight.get_tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    output = ColoTensor.from_torch_tensor(output_parallel, spec=output_spec)

    if weight.compute_spec.output_replicate:
        return output.to_replicate()
    else:
        return output


def colo_embedding_bag_1d(tp_mode: str,
                          input_tensor: ColoTensor,
                          weight: ColoTensor,
                          offsets: Optional[Tensor] = None,
                          max_norm: Optional[float] = None,
                          norm_type: float = 2,
                          scale_grad_by_freq: bool = False,
                          mode: str = "mean",
                          sparse: bool = False,
                          per_sample_weights: Optional[Tensor] = None,
                          include_last_offset: bool = False,
                          padding_idx: Optional[int] = None) -> ColoTensor:
    assert tp_mode in ('col',)
    funcs = {'col': colo_embedding_bag_1Dcol}
    return funcs[tp_mode](input_tensor,
                          weight,
                          offsets=offsets,
                          max_norm=max_norm,
                          norm_type=norm_type,
                          scale_grad_by_freq=scale_grad_by_freq,
                          mode=mode,
                          sparse=sparse,
                          per_sample_weights=per_sample_weights,
                          include_last_offset=include_last_offset,
                          padding_idx=padding_idx)


@colo_op_impl(F.embedding_bag)
def colo_embedding_bag(input_tensor: GeneralTensor,
                       weight: GeneralTensor,
                       offsets: Optional[Tensor] = None,
                       max_norm: Optional[float] = None,
                       norm_type: float = 2,
                       scale_grad_by_freq: bool = False,
                       mode: str = "mean",
                       sparse: bool = False,
                       per_sample_weights: Optional[Tensor] = None,
                       include_last_offset: bool = False,
                       padding_idx: Optional[int] = None):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.
    This method looks up an embedding table.
    """
    assert isinstance(weight, ColoTensor)
    input_tensor = convert_to_colo_tensor(input_tensor, weight.get_process_group())

    # Handle differen parallel actions.

    if not weight.has_compute_spec():    # No Model Parallel Applied
        assert weight.is_replicate(), 'Invalid weight spec for native embedding op'
        return ColoTensor.from_torch_tensor(tensor=F.embedding_bag(input_tensor,
                                                                   weight,
                                                                   offsets=offsets,
                                                                   max_norm=max_norm,
                                                                   norm_type=norm_type,
                                                                   scale_grad_by_freq=scale_grad_by_freq,
                                                                   mode=mode,
                                                                   sparse=sparse,
                                                                   per_sample_weights=per_sample_weights,
                                                                   include_last_offset=include_last_offset,
                                                                   padding_idx=padding_idx),
                                            spec=ColoTensorSpec(weight.get_process_group()))
    elif weight.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.is_shard_1dcol():
            tp_mode = 'col'
        else:
            raise NotImplementedError
        return colo_embedding_bag_1d(tp_mode,
                                     input_tensor,
                                     weight,
                                     offsets=offsets,
                                     max_norm=max_norm,
                                     norm_type=norm_type,
                                     scale_grad_by_freq=scale_grad_by_freq,
                                     mode=mode,
                                     sparse=sparse,
                                     per_sample_weights=per_sample_weights,
                                     include_last_offset=include_last_offset,
                                     padding_idx=padding_idx)
    else:
        raise NotImplementedError

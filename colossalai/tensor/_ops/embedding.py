import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.context import ParallelMode
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, \
    gather_forward_split_backward, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, dist_spec


def colo_embedding_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, args, kwargs) -> ColoTensor:
    # embedding_1Dcol split the weight(lookup table) to (num_embeddings, embedding_dim/P)
    # Gather splitted lookup table
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    input_tensor.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))

    output_parallel = torch.nn.functional.embedding(input_tensor.torch_tensor(), weight.torch_tensor(), *args, **kwargs)
    output_spec = TensorSpec(
        dist_spec.shard(weight.spec.get_process_group(), [-1], [weight.spec.get_process_group().size()]),
        [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)])
    output = ColoTensor.init_from_torch_tensor(output_parallel, spec=output_spec)
    output.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))
    return output


def colo_embedding_1Drow(input_tensor: ColoTensor, weight: ColoTensor, args, kwargs) -> ColoTensor:
    # embedding_1Drow split the weight(lookup table) to (num_embeddings/P, embedding_dim)
    # Find index in this shard and mask those not here
    # Reduce all
    parallel_action = weight.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    input_tensor.to_dist_spec(dist_spec.replicate(weight.spec.get_process_group()))

    tensor_parallel_rank = gpc.get_local_rank(parallel_action.parallel_mode)
    num_embeddings_per_partition = weight.size(0)
    vocab_start_index = tensor_parallel_rank * num_embeddings_per_partition
    vocab_end_index = vocab_start_index + num_embeddings_per_partition

    # Build the mask.
    input_mask = (input_tensor.torch_tensor() < vocab_start_index) | \
        (input_tensor.torch_tensor() >= vocab_end_index)
    # Mask the input.
    # TODO(jzy) masked_input may be an activation managed by ColoTensor.
    masked_input = input_tensor.torch_tensor().clone() - vocab_start_index
    masked_input[input_mask] = 0

    partial_output = torch.nn.functional.embedding(masked_input, weight.torch_tensor(), *args, **kwargs)

    # Mask the output embedding.
    partial_output[input_mask, :] = 0.
    # Reduce across all the model parallel GPUs.
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    output = ColoTensor.init_from_torch_tensor(output,
                                               spec=TensorSpec(dist_spec.replicate(weight.spec.get_process_group())))
    return output


@colo_op_impl(torch.nn.functional.embedding)
def colo_embedding(types, args, kwargs, pg):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method looks up an embedding table.
    """
    input_tensor = args[0]
    weight = args[1]
    args = args[2:]

    if not isinstance(input_tensor, ColoTensor):
        input_tensor = ColoTensor.init_from_torch_tensor(input_tensor)

    if not isinstance(weight, ColoTensor):
        weight = ColoTensor.init_from_torch_tensor(weight)

    # Handle differen parallel actions.

    if not weight.has_spec():    # No Model Parallel Applied
        assert weight.spec.is_gathered(), 'Invalid weight spec for native embedding op'
        input_tensor = input_tensor.torch_tensor()
        weight = weight.torch_tensor()
        output = torch.nn.functional.embedding(input_tensor, weight, *args, **kwargs)
        return ColoTensor.init_from_torch_tensor(output)
    elif weight.spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if weight.spec.is_1D_row():
            return colo_embedding_1Drow(input_tensor, weight, args, kwargs)
        elif weight.spec.is_1D_col():
            return colo_embedding_1Dcol(input_tensor, weight, args, kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

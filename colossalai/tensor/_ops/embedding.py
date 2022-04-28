import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.context import ParallelMode
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, \
    gather_forward_split_backward, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from packaging import version
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, ShardPattern

def colo_embedding_1Dcol(input_tensor: ColoTensor, weight: ColoTensor, args, kwargs) -> ColoTensor:
    # embedding_1Dcol split the weight(lookup table)
    # Gather splitted lookup table
    parallel_action = weight.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DCol_Embedding)
    if not input_tensor.is_gathered():
        input_tensor.gather()

    output_parallel = torch.nn.functional.embedding(input_tensor.torch_tensor(), weight.torch_tensor(), 
        *args, **kwargs)
    output = ColoTensor.init_from_torch_tensor(output_parallel)
    out_parallel_action_list = [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)]
    output_spec = TensorSpec(out_parallel_action_list)
    output.set_spec(output_spec, shard=False)
    output.set_shard_pattern(ShardPattern.Col)
    output.gather()
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
    if not weight.has_spec(): # No Model Parallel Applied
        input_tensor = input_tensor.torch_tensor()
        weight = weight.torch_tensor()
        output = torch.nn.functional.embedding(input_tensor, weight, *args, **kwargs)
        return ColoTensor.init_from_torch_tensor(output)
    elif weight.shard_spec.num_action == 1: # Single Model Parallel Applied
        compute_patterns = weight.shard_spec.compute_patterns
        if ComputePattern.TP1DCol_Embedding in compute_patterns:
            return colo_embedding_1Dcol(input_tensor, weight, args, kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

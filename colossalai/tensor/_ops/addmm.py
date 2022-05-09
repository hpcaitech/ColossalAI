import torch
from typing import Union
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor, ShardPattern
from colossalai.tensor.graph import GraphOpNode, GraphGlobalEnv


def colo_addmm_1Drow(input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Union[int, float],
                     alpha: Union[int, float]) -> ColoTensor:
    parallel_action = mat2.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DRow_mm)
    # mat1:S[1] x mat2:S[0] = Output:P
    # beta * input + alpha * All-Reduce(Output) = res

    # mat1:S[1]
    if mat1.is_gathered():
        # Not splited yet.
        assert divide(mat1.shape[-1], gpc.tensor_parallel_size) == mat2.size(0), \
            'Invalid shapes in 1Drow forward: mat1={}, mat2={}. Expected last dim of input {}.'.format(
            mat1.shape, mat2.shape, mat2.size(0) * gpc.tensor_parallel_size)
        input_per_partition = split_forward_gather_backward(mat1.torch_tensor(), parallel_action.parallel_mode, dim=-1)
    elif mat1.shard_pattern == ShardPattern.Col:
        # Splited by 1Dcol
        assert mat1.shape[-1] == mat2.size(0), \
            'Invalid shapes in 1Drow forward: mat1={}, mat2={}. Expected last dim of input {}.'.format(
            mat1.shape, mat2.shape, mat2.size(0))
        input_per_partition = mat1.torch_tensor()
    else:
        raise NotImplementedError

    # Output:P
    partial_output = torch.mm(input_per_partition, mat2.torch_tensor())
    # Reduce(Output)
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    # input
    assert not input_tensor.has_spec(), 'Invalid input spec for 1Drow addmm op'
    output = beta * input_tensor.torch_tensor() + alpha * output
    output = ColoTensor.init_from_torch_tensor(output)
    return output


def colo_addmm_1Dcol(input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Union[int, float],
                     alpha: Union[int, float]) -> ColoTensor:
    # mat1:B x mat2:S[1] + input:S[1] = Output:S[1]
    # All-Gather(Output)
    # mat1:B
    parallel_action = mat2.shard_spec.get_action_by_compute_pattern(ComputePattern.TP1DCol_mm)
    if mat1.is_gathered():
        # Not splited yet.
        assert mat1.shape[-1] == mat2.size(0), \
            'Invalid shapes in 1Dcol forward: mat1={}, mat2={}. Expected last dim of input {}.'.format(
                mat1.shape, mat2.shape, mat2.size(0))
        input_parallel = reduce_grad(mat1.torch_tensor(), parallel_action.parallel_mode)

    # input:S[1]
    assert input_tensor.has_spec() and input_tensor.shard_spec.num_action == 1 and \
        input_tensor.shard_pattern in [ShardPattern.Col, ShardPattern.Row], \
        'Invalid bias spec for 1Dcol Linear op'

    output_parallel = torch.addmm(input_tensor.torch_tensor(),
                                  input_parallel,
                                  mat2.torch_tensor(),
                                  beta=beta,
                                  alpha=alpha)

    output = ColoTensor.init_from_torch_tensor(output_parallel)
    out_parallel_action_list = [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)]
    output_spec = TensorSpec(out_parallel_action_list)
    output.set_spec(output_spec, shard=False)
    output.set_shard_pattern(ShardPattern.Col)
    if parallel_action.gather_out:
        # All-Gather(Output)
        output.gather()
    return output


@colo_op_impl(torch.addmm)
def colo_addmm(types, args, kwargs, pg):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    input_tensor, mat1, mat2 = tuple(
        map(lambda t: t if isinstance(t, ColoTensor) else ColoTensor.init_from_torch_tensor(t), args[:3]))
    beta = kwargs.get('beta', 1) if kwargs else 1
    alpha = kwargs.get('alpha', 1) if kwargs else 1

    # building the computing graph, inputs -> op
    # if GraphGlobalEnv().graph_building:
    #     cur_op_node = GraphOpNode('linear', [weight, bias])
    #     cur_op_node.add_prev_tensor(input_tensor)

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not mat2.has_spec():    # No Model Parallel Applied
        assert not input_tensor.has_spec(), 'Invalid input spec for native addmm op'
        ret_tensor = ColoTensor.init_from_torch_tensor(
            torch.addbmm(input_tensor.torch_tensor(), mat1.torch_tensor(), mat2.torch_tensor(), beta=beta, alpha=alpha))
    elif mat2.shard_spec.num_action == 1:    # Single Model Parallel Applied
        compute_patterns = mat2.shard_spec.compute_patterns
        if ComputePattern.TP1DRow_mm in compute_patterns:
            ret_tensor = colo_addmm_1Drow(input_tensor, mat1, mat2, beta, alpha)
        elif ComputePattern.TP1DCol_mm in compute_patterns:
            ret_tensor = colo_addmm_1Dcol(input_tensor, mat1, mat2, beta, alpha)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # building the computing graph, op -> output
    # if GraphGlobalEnv().graph_building:
    #     cur_op_node.add_post_tensor(ret_tensor)

    return ret_tensor

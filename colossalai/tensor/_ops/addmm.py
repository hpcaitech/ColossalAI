import torch
from typing import Union
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.nn.layer.parallel_1d._utils import split_forward_gather_backward, reduce_input, reduce_grad
from colossalai.nn.layer.utils import divide
from colossalai.core import global_context as gpc
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ParallelAction, ColoTensor
from colossalai.tensor.graph import GraphOpNode, GraphGlobalEnv
from colossalai.tensor import dist_spec


def colo_addmm_1Drow(input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Union[int, float],
                     alpha: Union[int, float]) -> ColoTensor:
    parallel_action = mat2.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    # mat1:S[1] x mat2:S[0] = Output:P
    # beta * input + alpha * All-Reduce(Output) = res

    mat1.to_dist_spec(dist_spec.shard(mat2.spec.get_process_group(), [-1], [mat2.spec.get_process_group().size()]))

    # Output:P
    partial_output = torch.mm(mat1.torch_tensor(), mat2.torch_tensor())
    # Reduce(Output)
    output = reduce_input(partial_output, parallel_action.parallel_mode)
    # input
    assert not input_tensor.has_spec(), 'Invalid input spec for 1Drow addmm op'
    output = beta * input_tensor.torch_tensor() + alpha * output
    output = ColoTensor.init_from_torch_tensor(output,
                                               spec=TensorSpec(dist_spec.replicate(mat2.spec.get_process_group())))
    return output


def colo_addmm_1Dcol(input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Union[int, float],
                     alpha: Union[int, float]) -> ColoTensor:
    # mat1:B x mat2:S[1] + input:S[1] = Output:S[1]
    parallel_action = mat2.spec.get_action_by_compute_pattern(ComputePattern.TP1D)
    mat1.to_dist_spec(dist_spec.replicate(mat2.spec.get_process_group()))
    mat1_torch_tensor = reduce_grad(mat1.torch_tensor(), parallel_action.parallel_mode)

    output_parallel = torch.addmm(input_tensor.torch_tensor(),
                                  mat1_torch_tensor,
                                  mat2.torch_tensor(),
                                  beta=beta,
                                  alpha=alpha)
    output_spec = TensorSpec(
        dist_spec.shard(mat2.spec.get_process_group(), [-1], [mat2.spec.get_process_group().size()]),
        [ParallelAction(priority=1, parallel_mode=parallel_action.parallel_mode)])
    output = ColoTensor.init_from_torch_tensor(output_parallel, spec=output_spec)
    if parallel_action.gather_out:
        # All-Gather(Output)
        output.to_dist_spec(dist_spec.replicate(mat2.spec.get_process_group()))
    return output


@colo_op_impl(torch.addmm)
def colo_addmm(types, args, kwargs, pg):
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    input_tensor, mat1, mat2 = args[:3]
    to_colo_tensor = lambda t: t if isinstance(t, ColoTensor) else ColoTensor.init_from_torch_tensor(t)
    input_tensor = to_colo_tensor(input_tensor)
    mat2 = to_colo_tensor(mat2)
    beta = kwargs.get('beta', 1) if kwargs else 1
    alpha = kwargs.get('alpha', 1) if kwargs else 1

    # building the computing graph, inputs -> op
    # if GraphGlobalEnv().graph_building:
    #     cur_op_node = GraphOpNode('linear', [weight, bias])
    #     cur_op_node.add_prev_tensor(input_tensor)

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not mat2.has_spec():    # No Model Parallel Applied
        assert mat2.spec.is_gathered(), 'Invalid mat2 spec for native addmm op'
        assert input_tensor.spec.is_gathered(), 'Invalid input spec for native addmm op'
        ret_tensor = ColoTensor.init_from_torch_tensor(
            torch.addmm(input_tensor.torch_tensor(), mat1, mat2.torch_tensor(), beta=beta, alpha=alpha))
    elif mat2.spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        spec = TensorSpec(dist_spec.replicate(mat2.spec.get_process_group()))
        mat1 = args[1] if isinstance(args[1], ColoTensor) else ColoTensor.init_from_torch_tensor(args[1], spec=spec)
        if mat2.spec.is_1D_row() and input_tensor.spec.is_gathered():
            ret_tensor = colo_addmm_1Drow(input_tensor, mat1, mat2, beta, alpha)
        elif mat2.spec.is_1D_col() and (input_tensor.spec.is_1D_col() or input_tensor.spec.is_1D_row()):
            ret_tensor = colo_addmm_1Dcol(input_tensor, mat1, mat2, beta, alpha)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # building the computing graph, op -> output
    # if GraphGlobalEnv().graph_building:
    #     cur_op_node.add_post_tensor(ret_tensor)

    return ret_tensor

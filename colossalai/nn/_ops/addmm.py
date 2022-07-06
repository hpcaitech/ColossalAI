import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ComputePattern, TensorSpec, ComputePattern, ComputeSpec, ColoTensor
from colossalai.tensor import distspec
from ._utils import GeneralTensor, Number, convert_to_colo_tensor
from ._utils import reduce_input, reduce_grad


def colo_addmm_1Drow(input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Number,
                     alpha: Number) -> ColoTensor:
    # mat1:S[1] x mat2:S[0] = Output:P
    # beta * input + alpha * All-Reduce(Output) = res

    mat1 = mat1.convert_to_dist_spec(distspec.shard(mat2.get_process_group(), [-1], [mat2.get_tp_world_size()]))

    # Output:P
    partial_output = torch.mm(mat1, mat2)
    # Reduce(Output)
    output = reduce_input(partial_output, mat1.get_process_group())
    # input
    assert not input_tensor.has_compute_spec(), 'Invalid input spec for 1Drow addmm op'
    output = beta * input_tensor + alpha * output
    output = ColoTensor.from_torch_tensor(output, spec=TensorSpec(distspec.replicate(mat2.get_process_group())))
    return output


def colo_addmm_1Dcol(input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Number,
                     alpha: Number) -> ColoTensor:
    # mat1:B x mat2:S[1] + input:S[1] = Output:S[1]
    compute_spec = mat2.tensor_spec.compute_spec
    mat1 = mat1.convert_to_dist_spec(distspec.replicate(mat2.get_process_group()))
    mat1 = reduce_grad(mat1, mat1.get_process_group())

    output_parallel = torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha)
    output_spec = TensorSpec(distspec.shard(mat2.get_process_group(), [-1], [mat2.get_tp_world_size()]),
                             ComputeSpec(ComputePattern.TP1D))
    output = ColoTensor.from_torch_tensor(output_parallel, spec=output_spec)

    if compute_spec.output_replicate:
        return output.to_replicate()
    else:
        return output


def colo_addmm_1d(mode: str, input_tensor: ColoTensor, mat1: ColoTensor, mat2: ColoTensor, beta: Number,
                  alpha: Number) -> ColoTensor:
    assert mode in ('row', 'col')
    funcs = {'row': colo_addmm_1Drow, 'col': colo_addmm_1Dcol}
    return funcs[mode](input_tensor, mat1, mat2, beta, alpha)


@colo_op_impl(torch.addmm)
def colo_addmm(input_tensor: GeneralTensor,
               mat1: GeneralTensor,
               mat2: GeneralTensor,
               *args,
               beta: Number = 1,
               alpha: Number = 1) -> ColoTensor:
    """Handles ``__torch_function__`` dispatch for ``torch.nn.functional.linear``.
    This method computes a linear.
    """
    input_tensor, mat1, mat2 = tuple(map(convert_to_colo_tensor, (input_tensor, mat1, mat2)))

    # Add communication logic before and after linear call.
    ret_tensor = None
    if not mat2.has_compute_spec():    # No Model Parallel Applied
        assert mat2.tensor_spec.is_replicate(), 'Invalid mat2 spec for native addmm op'
        assert input_tensor.tensor_spec.is_replicate(), 'Invalid input spec for native addmm op'
        ret_tensor = ColoTensor.from_torch_tensor(torch.addmm(input_tensor, mat1, mat2, beta=beta, alpha=alpha))
    elif mat2.tensor_spec.has_compute_pattern(ComputePattern.TP1D):    # Single Model Parallel Applied
        if mat2.tensor_spec.is_shard_1drow() and input_tensor.tensor_spec.is_replicate():
            mode = 'row'
        elif mat2.tensor_spec.is_shard_1dcol() and (input_tensor.tensor_spec.is_shard_1dcol()
                                                    or input_tensor.tensor_spec.is_shard_1drow()):
            mode = 'col'
        else:
            raise NotImplementedError
        ret_tensor = colo_addmm_1d(mode, input_tensor, mat1, mat2, beta, alpha)
    else:
        raise NotImplementedError

    return ret_tensor

from copy import copy
import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor
from ._utils import GeneralTensor


def register_elementwise_op(op):

    @colo_op_impl(op)
    def elementwise_op(input_tensor: GeneralTensor, *args, **kwargs):
        """
        Handles ``__torch_function__`` dispatch for the elementwise op such
        as ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.
        This method computes on either a normal tensor or a sharded tensor.
        """
        output = op(input_tensor, *args, **kwargs)
        if isinstance(input_tensor, ColoTensor):
            spec = copy(input_tensor.spec)
            return ColoTensor.from_torch_tensor(output, spec=spec)
        return ColoTensor.from_torch_tensor(output)


register_elementwise_op(torch.nn.functional.gelu)
register_elementwise_op(torch.nn.functional.relu)
register_elementwise_op(torch.clone)
register_elementwise_op(torch.Tensor.clone)
register_elementwise_op(torch.Tensor.detach)

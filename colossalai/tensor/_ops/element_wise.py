import torch
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor


@colo_op_impl(torch.mean)
def colo_mean(types, args=(), kwargs=None, pg=None):
    stateful_tensor = args[0]
    return torch.mean(stateful_tensor.torch_tensor())


def register_elementwise_op(op):

    @colo_op_impl(op)
    def elementwise_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the elementwise op such
        as ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.
        This method computes on either a normal tensor or a sharded tensor.
        """
        input_tensor = args[0]
        # Validate types
        if not isinstance(input_tensor, ColoTensor):
            raise TypeError("input needs to be a ColoTensor")
        return op(input_tensor.torch_tensor())


register_elementwise_op(torch.nn.functional.gelu)
register_elementwise_op(torch.nn.functional.relu)

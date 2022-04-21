import torch
from colossalai.tensor.op_wrapper import colo_op_impl


def validate_param(param, param_name):
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")


@colo_op_impl(torch.nn.init.uniform_)
def colo_uniform(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensor in sharded_tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        sharded_tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, "kwargs")
    stateful_tensor = kwargs["tensor"]
    validate_param(stateful_tensor, "stateful_tensor")
    a = kwargs['a']
    validate_param(a, "a")
    b = kwargs['b']
    validate_param(b, "b")

    torch.nn.init.uniform_(stateful_tensor.torch_tensor(), a=a, b=b)
    return stateful_tensor

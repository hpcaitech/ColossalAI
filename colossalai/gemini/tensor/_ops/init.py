import torch
from colossalai.gemini.tensor import stateful_op_impl


def validate_param(param, param_name):
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")


@stateful_op_impl(torch.nn.init.uniform_)
def uniform_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensor in sharded_tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        sharded_tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, "kwargs")
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "sharded_tensor")
    a = kwargs['a']
    validate_param(a, "a")
    b = kwargs['b']
    validate_param(b, "b")

    torch.nn.init.uniform_(sharded_tensor.torch_tensor, a=a, b=b)
    return sharded_tensor

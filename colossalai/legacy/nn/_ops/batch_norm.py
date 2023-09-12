from typing import Optional

import torch.nn.functional as F

from colossalai.tensor import ColoTensor, ColoTensorSpec, ReplicaSpec
from colossalai.tensor.op_wrapper import colo_op_impl

from ._utils import GeneralTensor, convert_to_colo_tensor


@colo_op_impl(F.batch_norm)
def colo_batch_norm(
    input: GeneralTensor,
    running_mean: Optional[GeneralTensor],
    running_var: Optional[GeneralTensor],
    weight: Optional[GeneralTensor] = None,
    bias: Optional[GeneralTensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    assert isinstance(weight, ColoTensor)
    running_mean = running_mean.detach()
    running_var = running_var.detach()

    input = convert_to_colo_tensor(input, weight.get_process_group())
    bias = convert_to_colo_tensor(bias, weight.get_process_group())
    input = input.redistribute(ReplicaSpec())
    bias = bias.redistribute(ReplicaSpec())

    output = F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
    output = ColoTensor.from_torch_tensor(tensor=output, spec=ColoTensorSpec(pg=weight.get_process_group()))
    return output

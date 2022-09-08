from typing import List, Optional
import torch.nn.functional as F
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor, distspec, ColoTensorSpec, ReplicaSpec
from ._utils import GeneralTensor, convert_to_colo_tensor


@colo_op_impl(F.layer_norm)
def colo_layernorm(
    input_tensor: GeneralTensor,
    normalized_shape: List[int],
    weight: Optional[GeneralTensor] = None,
    bias: Optional[GeneralTensor] = None,
    eps: float = 1e-5,
):
    assert isinstance(weight, ColoTensor)
    input_tensor = convert_to_colo_tensor(input_tensor, weight.get_process_group())
    bias = convert_to_colo_tensor(bias, weight.get_process_group())
    input_tensor = input_tensor.redistribute(ReplicaSpec())

    output = F.layer_norm(input_tensor, normalized_shape, weight=weight, bias=bias, eps=eps)
    output = ColoTensor.from_torch_tensor(tensor=output,
                                          spec=ColoTensorSpec(pg=input_tensor.get_process_group(),
                                                              dist_attr=input_tensor.dist_spec))
    return output

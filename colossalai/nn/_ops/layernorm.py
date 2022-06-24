import torch
import torch.nn.functional as F
from typing import List, Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor, distspec
from ._utils import GeneralTensor, convert_to_colo_tensor


@colo_op_impl(F.layer_norm)
def colo_layernorm(
    input_tensor: GeneralTensor,
    normalized_shape: List[int],
    weight: Optional[GeneralTensor] = None,
    bias: Optional[GeneralTensor] = None,
    eps: float = 1e-5,
):
    input_tensor, weight, bias = tuple(map(convert_to_colo_tensor, (input_tensor, weight, bias)))

    # TODO (ver217): check dist spec
    input_tensor = input_tensor.convert_to_dist_spec(distspec.replicate(input_tensor.tensor_spec.get_process_group()))

    output = F.layer_norm(input_tensor, normalized_shape, weight=weight, bias=bias, eps=eps)
    output = ColoTensor.from_torch_tensor(output, input_tensor.tensor_spec)
    return output

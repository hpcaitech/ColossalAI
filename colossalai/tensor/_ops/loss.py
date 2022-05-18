import torch
import torch.nn.functional as F
from typing import Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor
from colossalai.nn.loss.loss_1d import VocabParallelCrossEntropyLoss1D
from ._utils import GeneralTensor, convert_to_colo_tensor


@colo_op_impl(F.cross_entropy)
def colo_cross_entropy(input_tensor: GeneralTensor,
                       target: GeneralTensor,
                       weight: Optional[GeneralTensor] = None,
                       size_average: Optional[bool] = None,
                       ignore_index: int = -100,
                       reduce: Optional[bool] = None,
                       reduction: str = "mean",
                       label_smoothing: float = 0.0):
    input_tensor, target, weight = tuple(map(convert_to_colo_tensor, (input_tensor, target, weight)))

    if input_tensor.spec.is_gathered():    # Input is gathered
        output = F.cross_entropy(input_tensor,
                                 target,
                                 weight=weight,
                                 size_average=size_average,
                                 ignore_index=ignore_index,
                                 reduce=reduce,
                                 reduction=reduction,
                                 label_smoothing=label_smoothing)
        return ColoTensor.from_torch_tensor(output)
    elif input_tensor.has_spec() and input_tensor.spec.num_action == 1:    # Single Model Parallel Applied
        if input_tensor.spec.is_1D_col():
            output = VocabParallelCrossEntropyLoss1D()(input_tensor, target)
            return ColoTensor.from_torch_tensor(output)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

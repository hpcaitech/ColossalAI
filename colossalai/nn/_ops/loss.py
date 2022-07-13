import torch
import torch.nn.functional as F
from typing import Optional
from colossalai.tensor.op_wrapper import colo_op_impl
from colossalai.tensor import ColoTensor, ColoTensorSpec
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
    assert isinstance(weight, ColoTensor) or isinstance(target, ColoTensor) or isinstance(input_tensor, ColoTensor)
    pg = input_tensor.get_process_group() if isinstance(input_tensor, ColoTensor) else isinstance(target, ColoTensor)
    weight = convert_to_colo_tensor(weight, pg)
    target = convert_to_colo_tensor(target, pg)
    input_tensor = convert_to_colo_tensor(input_tensor, pg)

    if input_tensor.is_replicate():    # Input is gathered
        assert target.is_replicate() and (weight is None or weight.is_replicate()), \
            "Target tensor and weight tensor both should be complete"
        output = F.cross_entropy(input_tensor,
                                 target,
                                 weight=weight,
                                 size_average=size_average,
                                 ignore_index=ignore_index,
                                 reduce=reduce,
                                 reduction=reduction,
                                 label_smoothing=label_smoothing)
        return ColoTensor.from_torch_tensor(output, ColoTensorSpec(pg))
    elif input_tensor.has_compute_spec():    # Single Model Parallel Applied
        if input_tensor.is_shard_1dcol():
            assert weight is None, "Current TP cross entropy loss function doesn't support passing weight tensor in"
            assert target.is_replicate(), "Target tensor should be complete in TP cross entropy loss function"
            output = VocabParallelCrossEntropyLoss1D()(input_tensor,
                                                       target,
                                                       process_group=input_tensor.process_group.tp_process_group())
            return ColoTensor.from_torch_tensor(output, ColoTensorSpec(pg))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

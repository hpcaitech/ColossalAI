import torch
from typing import Union, Optional
from colossalai.tensor import ColoTensor

GeneralTensor = Union[ColoTensor, torch.Tensor]
Number = Union[int, float]


def convert_to_colo_tensor(tensor: Optional[GeneralTensor]) -> Optional[ColoTensor]:
    if tensor is not None and not isinstance(tensor, ColoTensor):
        tensor = ColoTensor.from_torch_tensor(tensor)
    return tensor

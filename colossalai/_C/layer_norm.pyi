from typing import List

from torch import Tensor

def forward_affine(input: Tensor, normalized_shape: List[int], gamma: Tensor, beta: Tensor, epsilon: float) -> List[Tensor]:
    ...


def backward_affine(dout: Tensor, mean: Tensor, invvar: Tensor, input: Tensor,
                    normalized_shape: List[int], gamma: Tensor, beta: Tensor, epsilon: float) -> List[Tensor]:
    ...

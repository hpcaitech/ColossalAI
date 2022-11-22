from torch import Tensor

def forward(input: Tensor, scale: float) -> Tensor:
    ...


def backward(output_grads: Tensor, softmax_results: Tensor, scale: float) -> Tensor:
    ...

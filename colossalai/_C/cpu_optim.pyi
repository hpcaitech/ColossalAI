from torch import Tensor

class CPUAdamOptimizer:
    def __init__(self, lr: float, beta1: float, beta2: float, eps: float,
                 weight_decay: float, adamw_mode: float) -> None: ...

    def step(self, step: int, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, bias_correction: bool,
             param: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor, loss_scale: float) -> None: ...

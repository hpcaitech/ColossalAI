from typing import List

from torch import Tensor

def multi_tensor_scale(chunk_size: int, noop_flag: Tensor, tensor_lists: List[List[Tensor]], scale: float) -> None:
    ...


def multi_tensor_sgd(chunk_size: int, noop_flag: Tensor, tensor_lists: List[List[Tensor]], weight_decay: float,
                     momentum: float, dampening: float, lr: float, nesterov: bool, first_run: bool, weight_decay_after_momentum: bool, scale: float) -> None:
    ...


def multi_tensor_adam(chunk_size: int, noop_flag: Tensor, tensor_lists: List[List[Tensor]], lr: float, beta1: float, beta2: float, epsilon: float, step: int, mode: int, bias_correction: int, weight_decay: float, div_scale: float) -> None:
    ...


def multi_tensor_lamb(chunk_size: int, noop_flag: Tensor, tensor_lists: List[List[Tensor]], lr: float, beta1: float, beta2: float, epsilon: float, step: int, bias_correction: int, weight_decay: float, grad_averaging: int, mode: int, global_grad_norm: Tensor, max_grad_norm: float, use_nvlamb_python: bool) -> None:
    ...


def multi_tensor_l2norm(chunk_size: int, noop_flag: Tensor, tensor_lists: List[List[Tensor]], per_tensor_python: bool) -> None:
    ...

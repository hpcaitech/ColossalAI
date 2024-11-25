from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, inf
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from colossalai.interface import OptimizerWrapper

from .mixed_precision_mixin import BF16MixedPrecisionMixin, FP16MixedPrecisionMixin


class NaiveFP16MixedPrecisionMixin(FP16MixedPrecisionMixin):
    def __init__(
        self,
        working_params: List[Parameter],
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
    ) -> None:
        super().__init__(
            initial_scale, min_scale, growth_factor, backoff_factor, growth_interval, hysteresis, max_scale
        )
        self.params = working_params

    def check_local_overflow(self) -> bool:
        for p in self.params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False


class MixedPrecisionOptimizer(OptimizerWrapper):
    def __init__(
        self,
        optim: Optimizer,
        precision: str = "fp16",
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
    ):
        super().__init__(optim)
        if precision == "fp16":
            working_params = []
            for group in self.optim.param_groups:
                for p in group["params"]:
                    working_params.append(p)
            self.mixed_precision = NaiveFP16MixedPrecisionMixin(
                working_params,
                initial_scale=initial_scale,
                min_scale=min_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                hysteresis=hysteresis,
                max_scale=max_scale,
            )
        elif precision == "bf16":
            self.mixed_precision = BF16MixedPrecisionMixin()
        else:
            raise ValueError(f"Unsupported precision: {precision}")
        self.max_norm = max_norm
        self.working_to_master_map: Dict[Parameter, Tensor] = {}
        self.master_to_working_map: Dict[Tensor, Parameter] = {}

        # create master weights
        for group in self.optim.param_groups:
            master_params = []
            for p in group["params"]:
                if p.requires_grad:
                    master_p = p
                    if p.dtype != torch.float:
                        master_p = p.detach().float()
                    self.working_to_master_map[p] = master_p
                    self.master_to_working_map[master_p] = p
                    master_params.append(master_p)
            group["params"] = master_params
        self._current_grad_norm: Optional[float] = None

    def backward(self, loss: Tensor, inputs=None, retain_graph=False, **kwargs):
        loss = self.mixed_precision.pre_backward(loss)
        loss.backward(inputs=inputs, retain_graph=retain_graph, **kwargs)

    def backward_by_grad(self, tensor: Tensor, grad: Tensor, inputs: Tensor = None, retain_graph: bool = False):
        grad = self.mixed_precision.pre_backward_by_grad(tensor, grad)
        torch.autograd.backward(
            tensors=tensor,
            grad_tensors=grad,
            inputs=inputs,
            retain_graph=retain_graph,
        )

    def zero_grad(self, *args, **kwargs):
        for p in self.working_to_master_map.keys():
            p.grad = None
        self.mixed_precision.pre_zero_grad()
        return super().zero_grad(*args, **kwargs)

    def _unscale_and_clip_grads(self, total_norm: float) -> None:
        """
        Unscale and clip gradients before performing the optimization step.

        Args:
            total_norm (float): The computed total gradient norm.

        Returns:
            None
        """
        div_scale = 1.0

        # If mixed-precision training is used, get the gradient division scale from the mixed-precision handler.
        if self.mixed_precision is not None:
            div_scale = self.mixed_precision.get_grad_div_scale()

        if self.max_norm > 0.0:
            # Calculate the scaling factor for gradient clipping
            # The gradient norm is scaled by 'div_scale' and then clipped to 'max_norm'
            clip = ((total_norm / div_scale) + 1e-6) / self.max_norm

            # If the clip factor exceeds 1, adjust 'div_scale' accordingly to ensure clipping
            if clip > 1:
                div_scale = clip * div_scale

        # Apply the scaling factor to gradients
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data.mul_(1.0 / div_scale)

    def _compute_grad_norm(self, param_gradient_pairs: List[Tuple[Tensor]], norm_type: int = 2) -> int:
        r"""
        Compute and return the gradient norm for gradient clipping.

        Args:
            param_gradient_pairs (List[Tuple[Tensor]]): List of (parameter, gradient) pairs; gradients are used for norm calculation.
            norm_type (int, optional): Type of the norm used (e.g., 2 for L2 norm). Defaults to 2.

        Returns:
            float: The total norm of the given gradients.
        """

        if len(param_gradient_pairs) == 0:
            return 0.0

        # gradients used for norm calculation.
        gradients = [grad for param, grad in param_gradient_pairs]

        if norm_type == inf:
            total_norm = max(grad.data.abs().max() for grad in gradients)

        else:
            total_norm_exponentiated = 0.0
            for grad in gradients:
                total_norm_exponentiated += grad.data.double().norm(norm_type) ** norm_type
            total_norm = total_norm_exponentiated ** (1.0 / norm_type)

        return total_norm

    def step(self, *args, **kwargs):
        if self.mixed_precision.should_skip_step():
            self.zero_grad()
            return
        # prepare grads
        for group in self.optim.param_groups:
            for p in group["params"]:
                working_param = self.master_to_working_map[p]
                if p is working_param:
                    continue
                if working_param.grad is not None:
                    p.grad = working_param.grad.data.float()
                    working_param.grad = None

        # gradient unscale and clip.
        if self.max_norm <= 0:
            # no need to compute gradient norm.
            total_norm = 0.0
        else:
            # compute the total norm.
            param_gradient_pairs = [
                (self.master_to_working_map[p], p.grad)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            total_norm = self._compute_grad_norm(param_gradient_pairs)
            self._current_grad_norm = total_norm
        self._unscale_and_clip_grads(total_norm)

        self.optim.step(*args, **kwargs)
        # update working params
        for group in self.optim.param_groups:
            for p in group["params"]:
                working_param = self.master_to_working_map[p]
                if p is working_param:
                    continue
                working_param.data.copy_(p.data)

    def update_master_params(self, model: Module):
        # Update master params from working params
        with torch.no_grad():
            for p in model.parameters():
                if (p is None) or (p not in self.working_to_master_map):
                    continue
                master_param = self.working_to_master_map[p]
                master_param.data.copy_(p.data)

    def get_working_to_master_map(self) -> Dict[int, torch.Tensor]:
        return {id(working_p): master_p for working_p, master_p in self.working_to_master_map.items()}

    def get_master_to_working_map(self) -> Dict[int, torch.Tensor]:
        return {id(master_p): working_p for master_p, working_p in self.master_to_working_map.items()}

    def get_grad_norm(self, norm_type=2, **kwargs):
        return self._current_grad_norm

from .mixed_precision_base import MixedPrecision


class FP16NaiveMixedPrecision(MixedPrecision):
    """
    Precision for mixed precision training in FP16 using naive AMP.

    Args:
    log_num_zeros_in_grad(bool): return number of zeros in the gradients.
    initial_scale(int): initial scale of gradient scaler.
    growth_factor(int): the growth rate of loss scale.
    backoff_factor(float): the decrease rate of loss scale.
    hysteresis(int): delay shift in dynamic loss scaling.
    max_scale(int): maximum loss scale allowed.
    verbose(bool): if set to `True`, will print debug info.
    """

    def __init__(
        self,
        log_num_zeros_in_grad: bool,
        initial_scale: int,
        growth_factor: int,
        backoff_factor: float,
        hysteresis: int,
        max_scale: int,
        verbose: bool = None,
    ) -> None:
        pass

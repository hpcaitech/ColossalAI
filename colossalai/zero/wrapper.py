from copy import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from .gemini import GeminiDDP


def zero_model_wrapper(
    model: nn.Module, zero_stage: int = 1, gemini_config: Optional[Dict] = None, verbose: bool = False
):
    """This wrapper function is used to wrap your training model for ZeRO DDP.

    Example:

        >>> with ColoInitContext():
        >>>     my_model = Bert()
        >>> my_optim = SGD(my_model.parameters(), lr = 1e-3)
        >>> zero_model = zero_model_wrapper(my_model, zero_stage=1)
        >>> zero_optim = zero_optim_wrapper(zero_model, my_optim)

    Args:
        model (nn.Module): The model used in ZeRO DDP.
        zero_stage (int, optional): The stage of ZeRO DDP. You can find more information in ZeRO's paper.
            https://arxiv.org/abs/1910.02054
        gemini_config (dict, optional): The configuration dictionary of `GeminiDDP`. `GeminiDDP` is enabled
            when the stage is set to 3. You can set the arguments of `GeminiDDP` in the gemini_config.
            Here is an example where we set the device of the model, the placement policy of Gemini, and the
            size of hidden dimension to help Gemini find out a unified chunk size.

            Example:

                >>> config_dict = dict(device=torch.cuda.current_device(), hidden_dim=1024, placement_policy='auto')
                >>> model = zero_model_wrapper(model, zero_stage=3, gemini_config=config_dict)
    """
    assert zero_stage in [1, 2, 3], "The stage of ZeRO should be 1, 2 or 3"

    if gemini_config is None:
        gemini_config = dict()

    if zero_stage in [1, 2]:
        wrapped_model = model
    else:
        wrapped_model = GeminiDDP(model, **gemini_config, verbose=verbose)

    setattr(wrapped_model, "_colo_zero_stage", zero_stage)

    return wrapped_model


def zero_optim_wrapper(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    initial_scale: float = 2**16,
    growth_factor: float = 2,
    backoff_factor: float = 0.5,
    growth_interval: int = 1000,
    hysteresis: int = 2,
    min_scale: float = 1,
    max_scale: float = 2**32,
    max_norm: float = 0.0,
    norm_type: float = 2.0,
    optim_config: Optional[Dict] = None,
    verbose: bool = False,
):
    """This wrapper function is used to wrap your training optimizer for ZeRO DDP.

    Args:
        model (nn.Module): Your model wrapped by `zero_model_wrapper`
        optimizer (torch.optim.Optimizer): Your initialized optimizer
        initial_scale (float, optional): initial_scale used by DynamicGradScaler.
        min_scale (float, optional): min_scale used by DynamicGradScaler.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler.
        max_scale (int, optional): max_scale used by DynamicGradScaler.
        max_norm (float, optional): max_norm used for `clip_grad_norm`. You should notice that you shall not do
            clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.
        norm_type (float, optional): norm_type used for `clip_grad_norm`.
        optim_config (dict, optional): The configuration used for the ZeRO optimizer.
            Example:

                >>> zero2_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True)
                >>> optim = zero_optim_wrapper(model, optim, optim_config=zero2_config)
        verbose (bool, optional): Whether to print the verbose info.
    """
    assert hasattr(model, "_colo_zero_stage"), "You should use `zero_ddp_wrapper` first"
    zero_stage = getattr(model, "_colo_zero_stage")

    assert norm_type == 2.0, "Current ZeRO optimizers only support 'norm_type=2'"

    if optim_config is None:
        config_dict = dict()
    else:
        config_dict = copy(optim_config)

    config_dict["initial_scale"] = initial_scale
    config_dict["growth_factor"] = growth_factor
    config_dict["backoff_factor"] = backoff_factor
    config_dict["growth_interval"] = growth_interval
    config_dict["hysteresis"] = hysteresis
    config_dict["min_scale"] = min_scale
    config_dict["max_scale"] = max_scale

    if zero_stage in [1, 2]:
        from colossalai.zero.low_level import LowLevelZeroOptimizer

        config_dict["partition_grad"] = zero_stage == 2
        config_dict["clip_grad_norm"] = max_norm
        return LowLevelZeroOptimizer(optimizer, **config_dict, verbose=verbose)
    else:
        from colossalai.zero.gemini.gemini_optimizer import GeminiOptimizer

        config_dict["clipping_norm"] = max_norm
        return GeminiOptimizer(optimizer, model, **config_dict, verbose=verbose)

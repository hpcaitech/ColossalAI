from typing import Optional

from .mixed_precision_base import MixedPrecision


class FP16ApexMixedPrecision(MixedPrecision):
    """
    Precision for mixed precision training in FP16 using PyTorch AMP.

    Args:
        opt_level(str, optional, default="O1" ): Pure or mixed precision optimization level. Accepted values are “O0”, “O1”, “O2”, and “O3”, explained in detail above Apex AMP Documentation.
        num_losses(int, optional, default=1): Option to tell AMP in advance how many losses/backward passes you plan to use. When used in conjunction with the loss_id argument to `amp.scale_loss`, enables Amp to use a different loss scale per loss/backward pass, which can improve stability. If num_losses is left to 1, Amp will still support multiple losses/backward passes, but use a single global loss scale for all of them.
        verbosity(int, default=1): Set to 0 to suppress Amp-related output.
        min_loss_scale(float, default=None): Sets a floor for the loss scale values that can be chosen by dynamic loss scaling. The default value of None means that no floor is imposed. If dynamic loss scaling is not used, min_loss_scale is ignored.
        max_loss_scale(float, default=2.**24 ): Sets a ceiling for the loss scale values that can be chosen by dynamic loss scaling. If dynamic loss scaling is not used, max_loss_scale is ignored.
        **kwargs:Currently, the under-the-hood properties that govern pure or mixed precision training are the following:
            cast_model_type, patch_torch_functions, keep_batchnorm_fp32, master_weights, loss_scale.
            cast_model_type: Casts your model’s parameters and buffers to the desired type.patch_torch_functions: Patch all Torch functions and Tensor methods to perform Tensor Core-friendly ops like GEMMs and convolutions in FP16, and any ops that benefit from FP32 precision in FP32.
            keep_batchnorm_fp32: To enhance precision and enable cudnn batchnorm (which improves performance), it’s often beneficial to keep batchnorm weights in FP32 even if the rest of the model is FP16.
            master_weights: Maintain FP32 master weights to accompany any FP16 model weights. FP32 master weights are stepped by the optimizer to enhance precision and capture small gradients.
            loss_scale: If loss_scale is a float value, use this value as the static (fixed) loss scale. If loss_scale is the string "dynamic", adaptively adjust the loss scale over time. Dynamic loss scale adjustments are performed by Amp automatically.
    """

    def __init__(
        self,
        opt_level: Optional[str] = "O1",
        num_losses: Optional[int] = 1,
        verbosity: int = 1,
        min_loss_scale: float = None,
        max_loss_scale: float = 2.**24,
        **kwargs,
    ) -> None:
        pass

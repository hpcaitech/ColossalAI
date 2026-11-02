"""How the reference reduces precision to match a plugin configuration (precision recipes) and which configurations
are supported. Used by ``training.run_reference`` and by the independent-implementation scenario."""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from colossalai.booster.mixed_precision import FP16TorchMixedPrecision, MixedPrecision
from colossalai.booster.plugin import LowLevelZeroPlugin, Plugin, TorchDDPPlugin


@dataclass(frozen=True)
class Recipe:
    """How the reference reduces precision to match a plugin configuration, in plain PyTorch.

    Args:
        dtype: The compute dtype.
        autocast: If True, weights stay float32 and the forward runs under ``torch.autocast(dtype)``. Otherwise the
            whole model and the floating-point inputs are cast to ``dtype``.
        master_weights: Keep a float32 copy of the cast weights; gradients are cast to float32, the optimizer updates
            the copy and the result is cast back.
        loss_scale: Fixed factor the loss is multiplied by before backward; gradients are divided by it afterwards.
        source: Where the plugin does this, so that a change on its side can be traced back here.
    """

    dtype: torch.dtype
    autocast: bool = False
    master_weights: bool = False
    loss_scale: Optional[float] = None
    source: str = ""


FLOAT64 = Recipe(torch.float64, source="float64 ground truth")


class UnsupportedRecipe(ValueError):
    pass


def recipe_for(plugin: Plugin, mixed_precision: Union[str, MixedPrecision, None] = None) -> Recipe:
    """The reference recipe matching ``Booster(plugin=plugin, mixed_precision=mixed_precision)``.

    Only configurations whose recipe has been checked against the plugin source are listed. Anything else raises
    ``UnsupportedRecipe``: a reference with a different recipe would make the comparison meaningless.
    """
    if isinstance(plugin, LowLevelZeroPlugin):
        if mixed_precision is not None:
            raise UnsupportedRecipe("LowLevelZeroPlugin controls precision itself; pass precision= to the plugin")
        kwargs = plugin.zero_optim_kwargs
        if kwargs["clip_grad_norm"] > 0 or not plugin.cast_inputs or plugin.use_fp8 or kwargs["cpu_offload"]:
            raise UnsupportedRecipe(
                "LowLevelZeroPlugin with max_norm, cast_inputs=False, use_fp8 or cpu_offload is not verified"
            )
        if plugin.precision == "fp32":
            return Recipe(torch.float32, source="LowLevelZeroModel leaves a float32 module unchanged")
        if plugin.precision == "bf16":
            return Recipe(
                torch.bfloat16,
                master_weights=kwargs["master_weights"],
                source="low_level_zero_plugin.py LowLevelZeroModel: module.to(bf16), forward casts floating inputs; "
                "low_level_optim.py step: gradients cast to float32 master weights, no loss scaling for bf16",
            )
        raise UnsupportedRecipe(f"LowLevelZeroPlugin precision={plugin.precision!r} is not verified")

    if isinstance(plugin, TorchDDPPlugin):
        if mixed_precision is None or mixed_precision == "fp32":
            return Recipe(torch.float32, source="TorchDDPPlugin does not change dtypes")
        if mixed_precision == "fp16":
            mixed_precision = FP16TorchMixedPrecision()
        if isinstance(mixed_precision, FP16TorchMixedPrecision):
            return Recipe(
                torch.float16,
                autocast=True,
                loss_scale=mixed_precision.torch_amp_kwargs["init_scale"],
                source="fp16_torch.py: TorchAMPModule runs forward under autocast; TorchAMPOptimizer scales the loss "
                "with a GradScaler starting at init_scale, which stays constant over a short run without overflow",
            )
        if mixed_precision == "bf16":
            raise UnsupportedRecipe("TorchDDPPlugin with bf16 cannot be built: BF16MixedPrecision is not implemented")
        raise UnsupportedRecipe(f"TorchDDPPlugin with mixed_precision={mixed_precision!r} is not verified")

    raise UnsupportedRecipe(
        f"{type(plugin).__name__} is not supported yet; supported: TorchDDPPlugin, LowLevelZeroPlugin"
    )

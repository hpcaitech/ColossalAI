"""How the reference reduces precision to match a plugin configuration (precision recipes) and which configurations
are supported. Used by ``training.run_reference`` and by the independent-implementation scenario."""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from colossalai.accelerator import get_accelerator
from colossalai.booster.mixed_precision import BF16MixedPrecision, FP16TorchMixedPrecision, MixedPrecision
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, Plugin, TorchDDPPlugin
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin


@dataclass(frozen=True)
class Recipe:
    """How the reference reduces precision to match a plugin configuration, in plain PyTorch.

    Args:
        dtype: The compute dtype.
        autocast: If True, weights stay float32 and the forward runs under ``torch.autocast(dtype)``. Otherwise the
            whole model and the floating-point inputs are cast to ``dtype``.
        master_weights: Keep a float32 copy of the cast weights; gradients are cast to float32, the optimizer updates
            the copy and the result is cast back.
        loss_scale: Fixed factor the loss is multiplied by before backward; gradients are divided by it afterwards, in
            float32 with ``master_weights`` (as they reach the master weights), otherwise in their own dtype.
        optimizer_device: Where the master weights and the optimizer live (``"cpu"`` for offloading plugins); ``None``
            keeps them on the accelerator. Only meaningful with ``master_weights``.
        master_from_float32: The master weights start from the float32 weights the model was built with, not from
            their cast to ``dtype`` (whose low bits are lost). Only meaningful with ``master_weights``.
        source: Where the plugin does this, so that a change on its side can be traced back here.
    """

    dtype: torch.dtype
    autocast: bool = False
    master_weights: bool = False
    loss_scale: Optional[float] = None
    optimizer_device: Optional[str] = None
    master_from_float32: bool = False
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
        if kwargs["clip_grad_norm"] > 0 or not plugin.cast_inputs or plugin.use_fp8:
            raise UnsupportedRecipe("LowLevelZeroPlugin with max_norm, cast_inputs=False or use_fp8 is not verified")
        offload = kwargs["cpu_offload"]
        if plugin.precision == "fp32":
            if offload:
                raise UnsupportedRecipe("LowLevelZeroPlugin with precision='fp32' and cpu_offload is not verified")
            return Recipe(torch.float32, source="LowLevelZeroModel leaves a float32 module unchanged")
        if plugin.precision == "bf16":
            if offload and not kwargs["master_weights"]:
                raise UnsupportedRecipe("LowLevelZeroPlugin with cpu_offload and master_weights=False is not verified")
            source = (
                "low_level_zero_plugin.py LowLevelZeroModel: module.to(bf16), forward casts floating inputs; "
                "low_level_optim.py step: gradients cast to float32 master weights, no loss scaling for bf16"
            )
            if offload:
                source += (
                    "; cpu_offload: master weights created on cpu, gradients moved there, the optimizer steps there"
                )
            return Recipe(
                torch.bfloat16,
                master_weights=kwargs["master_weights"],
                optimizer_device="cpu" if offload else None,
                source=source,
            )
        raise UnsupportedRecipe(f"LowLevelZeroPlugin precision={plugin.precision!r} is not verified")

    if isinstance(plugin, HybridParallelPlugin):
        # the shardformer path: the model is rewritten by its policy (kernel replacement included), with tp_size=1
        if mixed_precision is not None:
            raise UnsupportedRecipe("HybridParallelPlugin controls precision itself; pass precision= to the plugin")
        parallel = (plugin.tp_size, plugin.pp_size, plugin.sp_size, getattr(plugin, "ep_size", 1))
        if any(size > 1 for size in parallel) or plugin.enable_sequence_parallelism:
            raise UnsupportedRecipe(
                "HybridParallelPlugin with tensor, pipeline, sequence or expert parallelism is not verified; "
                "use tp_size=1, pp_size=1"
            )
        if plugin.zero_stage > 0 or plugin.max_norm > 0 or plugin.use_fp8 or plugin.cpu_offload:
            raise UnsupportedRecipe(
                "HybridParallelPlugin with zero_stage > 0, max_norm, use_fp8 or cpu_offload is not verified"
            )
        if plugin.precision == "fp32":
            return Recipe(
                torch.float32, source="HybridParallelModule leaves a float32 module unchanged; plain optimizer"
            )
        if plugin.precision == "bf16":
            return Recipe(
                torch.bfloat16,
                master_weights=True,
                source="hybrid_parallel_plugin.py HybridParallelModule: module.to(bf16), floating inputs cast; "
                "HybridParallelAMPOptimizer (mixed_precision_optimizer.py): float32 master weights, no loss "
                "scaling for bf16",
            )
        raise UnsupportedRecipe(f"HybridParallelPlugin precision={plugin.precision!r} is not verified")

    if isinstance(plugin, GeminiPlugin):
        if mixed_precision is not None:
            raise UnsupportedRecipe("GeminiPlugin controls precision itself; pass precision= to the plugin")
        config, optim = plugin.gemini_config, plugin.optim_kwargs
        if plugin.tp_size > 1 or plugin.extra_dp_size > 1:
            raise UnsupportedRecipe("GeminiPlugin with tp_size or extra_dp_size > 1 is not verified")
        offloaded = (
            config["placement_policy"] != "static" or config["offload_optim_frac"] or config["offload_param_frac"]
        )
        if offloaded or optim["max_norm"] > 0 or config["use_fp8"] or not config["master_weights"]:
            raise UnsupportedRecipe(
                "GeminiPlugin with offloading (placement_policy='auto', offload_*_frac), max_norm, use_fp8 or "
                "master_weights=False is not verified"
            )
        dtype = config["mixed_precision"]
        source = (
            "gemini_ddp.py GeminiDDP: parameters cast to the chunk dtype, forward casts floating inputs; "
            "gemini_optimizer.py: float32 master weights in chunks"
        )
        if dtype == torch.bfloat16:
            return Recipe(torch.bfloat16, master_weights=True, source=source + ", no loss scaling for bf16")
        return Recipe(
            torch.float16,
            master_weights=True,
            loss_scale=optim["initial_scale"],
            source=source + "; GeminiFP16MixedPrecisionMixin scales the loss with a dynamic scaler starting at "
            "initial_scale, which stays constant over a short run without overflow; GeminiOptimizer divides it out "
            "in float32, in the optimizer step (div_scale)",
        )

    if isinstance(plugin, TorchFSDPPlugin):
        if mixed_precision is not None:
            raise UnsupportedRecipe("TorchFSDPPlugin controls precision itself; pass mixed_precision= to the plugin")
        kwargs = plugin.fsdp_kwargs
        unverified = [k for k in ("cpu_offload", "auto_wrap_policy", "ignored_modules") if kwargs[k] is not None]
        if unverified or plugin.fp8_communication:
            raise UnsupportedRecipe(
                f"TorchFSDPPlugin with {', '.join(unverified) or 'fp8_communication'} is not verified"
            )
        mp = kwargs["mixed_precision"]
        if mp is None:
            return Recipe(torch.float32, source="FSDP without MixedPrecision keeps float32 parameters")
        dtype = mp.param_dtype
        # buffer_dtype=None leaves buffers (e.g. rotary frequencies) in float32, unlike the reference's whole-model cast
        same = mp.buffer_dtype == dtype and mp.reduce_dtype in (None, dtype)
        if dtype not in (torch.bfloat16, torch.float16) or not same or mp.keep_low_precision_grads:
            raise UnsupportedRecipe(
                "TorchFSDPPlugin is verified with MixedPrecision(param_dtype=bf16 or fp16) whose buffer_dtype equals "
                "param_dtype, reduce_dtype is unset or equal to it, and keep_low_precision_grads=False"
            )
        return Recipe(
            dtype,
            master_weights=True,
            master_from_float32=True,
            source="FSDP MixedPrecision: sharded float32 parameters cast to param_dtype for forward and backward, "
            "root forward casts floating inputs, gradients cast back to float32 for the optimizer; the sharded "
            "parameters are the model's own float32 weights, so the master weights start unrounded; "
            "torch_fsdp_plugin.py applies no loss scaling",
        )

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
        if mixed_precision == "bf16" or isinstance(mixed_precision, BF16MixedPrecision):
            if get_accelerator().name == "cpu":
                raise UnsupportedRecipe(
                    "TorchDDPPlugin with bf16 needs a CUDA or NPU device: CPUAccelerator.autocast does not "
                    "implement BF16 autocast"
                )
            return Recipe(
                torch.bfloat16,
                autocast=True,
                source="bf16.py BF16Module runs forward under accelerator BF16 autocast; parameters, gradients "
                "and optimizer states stay float32, with no loss scaling",
            )
        raise UnsupportedRecipe(f"TorchDDPPlugin with mixed_precision={mixed_precision!r} is not verified")

    raise UnsupportedRecipe(
        f"{type(plugin).__name__} is not supported yet; "
        "supported: TorchDDPPlugin, LowLevelZeroPlugin, HybridParallelPlugin (tp_size=1), GeminiPlugin, TorchFSDPPlugin"
    )

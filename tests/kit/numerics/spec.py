"""Per-model settings for whole-model alignment, attached to a model_zoo entry through ``ModelAttribute.numerics``."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from torch import nn


@dataclass(frozen=True)
class NumericsSpec:
    """How to check a model_zoo entry against its reference implementation.

    An entry takes part in whole-model alignment only if its ``ModelAttribute`` carries a ``NumericsSpec``;
    ``NumericsSpec()`` with no arguments is enough for a HuggingFace model.

    The model runs in train mode with every dropout probability set to 0. ``data_gen_fn`` is called once per step and
    the same batches are fed to every run, so random inputs (noise, timesteps, masks) must be produced there, not inside
    ``forward``. The reference also runs in float64, so its config must select implementations that support it, e.g.
    ``attn_implementation="eager"`` and, for HuggingFace MoE models, ``experts_implementation="eager"`` (the default
    grouped_mm kernel has no float64).

    Args:
        reference_fn: Builds the reference module, e.g. a wrapper around an official repository's model. It must accept
            the keyword arguments returned by ``data_gen_fn`` and return its outputs in the same structure as the model
            under test. ``None`` (the default) uses ``model_fn`` itself, which is the reference for HuggingFace models.
        param_map: Maps a reference parameter name to the name of the same parameter in the model under test, or to
            ``None`` if it has no counterpart. Used to copy the weights into the reference and to match parameter
            gradients. Only needed together with ``reference_fn`` when the names differ.
        output_fn: Selects what to compare from ``output_transform_fn(output)``, returning a tensor or a nested
            dict / list / tuple of tensors. ``None`` compares every floating-point tensor in it.
    """

    reference_fn: Optional[Callable[[], nn.Module]] = None
    param_map: Optional[Callable[[str], Optional[str]]] = None
    output_fn: Optional[Callable[[Any], Any]] = None

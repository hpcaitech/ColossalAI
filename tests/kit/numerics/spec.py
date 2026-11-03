"""Per-model settings for whole-model alignment, attached to a model_zoo entry through ``ModelAttribute.numerics``."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from torch import Tensor, nn


def _first_output(output: Any) -> Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


@dataclass(frozen=True)
class Routers:
    """The modules of a mixture-of-experts model that route tokens to experts, to diagnose routing flips.

    At low precision a router can pick a different set of experts for a token than the float64 truth, and that token's
    outputs and gradients then differ far more than the baseline error. Each run records, for its first forward, the
    top-``top_k`` experts of every token at every router call; the report counts the tokens whose expert set differs
    from the truth's, in ``test`` and in ``ref``. The counts are diagnostics only and never change a verdict.

    Args:
        select: ``select(name, module)`` is True for a router, given qualified module names of the model being run
            (the model under test, the reference, or each side of ``check_fwd_bwd``). Router calls are matched between
            runs by their order, so the reference may use other names.
        top_k: Number of experts each token is sent to.
        scores_fn: Picks, from a router's output, the scores the model takes the top-k of, with the experts in the last
            dimension. Defaults to the output itself, or its first element for a tuple. Models that add a bias or
            limit the groups before selecting must return those adjusted scores.
    """

    select: Callable[[str, nn.Module], bool]
    top_k: int
    scores_fn: Callable[[Any], Tensor] = _first_output


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
        unused_params: Names (in the model under test) of trainable parameters that legitimately get no gradient from
            ``data_gen_fn``'s batches, e.g. a branch the inputs never reach. Any other parameter with
            ``requires_grad=True`` and no gradient fails the check; frozen parameters are not expected to have one.
        routers: For mixture-of-experts models: which modules route tokens, so the report counts routing flips (see
            ``Routers``).
    """

    reference_fn: Optional[Callable[[], nn.Module]] = None
    param_map: Optional[Callable[[str], Optional[str]]] = None
    output_fn: Optional[Callable[[Any], Any]] = None
    unused_params: Tuple[str, ...] = ()
    routers: Optional[Routers] = None

"""Per-model benchmark settings, attached to a model_zoo entry through ``ModelAttribute.benchmark``."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch
from torch import Tensor, nn


def causal_lm_batch(model: nn.Module, micro_batch: int, seq_len: int, generator: torch.Generator) -> Dict[str, Tensor]:
    """Random token ids over the model's whole vocabulary, with the inputs as labels (HuggingFace shifts them)."""
    input_ids = torch.randint(model.config.vocab_size, (micro_batch, seq_len), generator=generator)
    return dict(input_ids=input_ids, labels=input_ids.clone())


def count_input_ids(batch: Dict[str, Tensor]) -> int:
    return batch["input_ids"].numel()


def model_loss(output: Any) -> Tensor:
    return output.loss


@dataclass(frozen=True)
class BenchmarkSpec:
    """How to benchmark a model_zoo entry at its real sizes.

    The entry's own ``model_fn`` builds a reduced model for unit tests; ``models`` builds the sizes that are actually
    trained, with randomly initialized weights. Each step feeds one synthetic batch of fixed shape, built once before
    the first step, so data loading is not part of the measurement.

    Args:
        models: Size name (e.g. ``"8b"``) to a function that builds the model at that size, on the CPU, in float32.
            The first one is the default.
        seq_len: Default sequence length.
        micro_batch: Default number of samples per device per step.
        batch_fn: ``batch_fn(model, micro_batch, seq_len, generator)`` returns one batch for ``model(**batch)``, on the
            CPU. Defaults to random token ids with themselves as labels (``causal_lm_batch``).
        tokens_fn: ``tokens_fn(batch)`` is the number of tokens a batch contributes to throughput. Defaults to the
            number of input ids; a diffusion model counts latent tokens.
        loss_fn: ``loss_fn(output)`` is the scalar the step backpropagates. Defaults to ``output.loss``.
        prepare_fn: Called on every freshly built model before it is boosted, e.g. to freeze a tower. It may modify the
            module in place and return ``None``, or return a replacement.
        flops_fn: ``flops_fn(model, seq_len)`` is the model FLOPs per token from a closed formula, given the model built
            on the meta device, e.g. ``flops.decoder_flops_per_token``. Without one, MFU uses the FLOPs that
            ``FlopCounterMode`` counts; with one, both are reported and checked against each other (see ``flops.py``).
    """

    models: Mapping[str, Callable[[], nn.Module]]
    seq_len: int = 4096
    micro_batch: int = 1
    batch_fn: Callable[[nn.Module, int, int, torch.Generator], Dict[str, Any]] = causal_lm_batch
    tokens_fn: Callable[[Dict[str, Any]], int] = count_input_ids
    loss_fn: Callable[[Any], Tensor] = model_loss
    prepare_fn: Optional[Callable[[nn.Module], Optional[nn.Module]]] = None
    flops_fn: Optional[Callable[[nn.Module, int], float]] = None

    def __post_init__(self):
        if not self.models:
            raise ValueError("BenchmarkSpec.models needs at least one size")

    @property
    def default_size(self) -> str:
        return next(iter(self.models))

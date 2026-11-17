"""What a model line declares once per model: ``BenchmarkSpec``, attached to a model_zoo entry through
``ModelAttribute.benchmark``.

model_zoo imports this module, so it must not import ColossalAI. An implementation that swaps in ColossalAI modules
(``ColoAttention``, the MoE layer) imports them inside its function, which runs only when that implementation is
measured.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

# the model exactly as ``BenchmarkSpec.models`` builds it
DEFAULT_IMPLEMENTATION = "default"


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
        prepare_fn: Called on every freshly built model, after its implementation is applied, e.g. to freeze a tower.
            It may modify the module in place and return ``None``, or return a replacement.
        causal_attention: The model's self-attention is causal, so half of its score matrix is masked and not
            computed; counted FLOPs halve it (``metrics.compute.AttentionCounter``). True for decoder language models.
        flops_fn: ``flops_fn(model, seq_len)`` is the model FLOPs per token from a closed formula, given the model built
            on the meta device, e.g. ``metrics.compute.decoder_flops_per_token``. Without one, MFU uses the counted
            FLOPs; with one, both are reported and checked against each other.
        implementations: Other ways to compute the same model, by name, e.g. ``{"fused": use_fused_kernels}``. Each
            function takes the model as built (the ``"default"`` implementation) and swaps its modules, in place
            (returning ``None``) or by returning a replacement. Implementations compute the same FLOPs, so they share
            one FLOPs count.
        reference_implementation: The implementation whose FLOPs are counted. Counting runs the model on the meta
            device, which fused kernels (flash attention, a Triton kernel) and communicating layers (expert
            parallelism) cannot; the reference must be one that can, usually the plain PyTorch one.
    """

    models: Mapping[str, Callable[[], nn.Module]]
    seq_len: int = 4096
    micro_batch: int = 1
    batch_fn: Callable[[nn.Module, int, int, torch.Generator], Dict[str, Any]] = causal_lm_batch
    tokens_fn: Callable[[Dict[str, Any]], int] = count_input_ids
    loss_fn: Callable[[Any], Tensor] = model_loss
    prepare_fn: Optional[Callable[[nn.Module], Optional[nn.Module]]] = None
    causal_attention: bool = False
    flops_fn: Optional[Callable[[nn.Module, int], float]] = None
    implementations: Mapping[str, Callable[[nn.Module], Optional[nn.Module]]] = field(default_factory=dict)
    reference_implementation: str = DEFAULT_IMPLEMENTATION

    def __post_init__(self):
        if not self.models:
            raise ValueError("BenchmarkSpec.models needs at least one size")
        if DEFAULT_IMPLEMENTATION in self.implementations:
            raise ValueError(f"{DEFAULT_IMPLEMENTATION!r} is the model as built; name the other implementations")
        if self.reference_implementation not in self.implementation_names:
            raise ValueError(
                f"reference implementation {self.reference_implementation!r} is not one of {self.implementation_names}"
            )

    @property
    def default_size(self) -> str:
        return next(iter(self.models))

    @property
    def implementation_names(self) -> Tuple[str, ...]:
        return (DEFAULT_IMPLEMENTATION, *self.implementations)

    def build(self, size: str, implementation: str = DEFAULT_IMPLEMENTATION) -> nn.Module:
        """The model at ``size`` in ``implementation``, prepared; on the current default device."""
        if size not in self.models:
            raise ValueError(f"no size {size!r}; sizes: {list(self.models)}")
        if implementation not in self.implementation_names:
            raise ValueError(f"no implementation {implementation!r}; implementations: {self.implementation_names}")
        model = self.models[size]()
        if implementation != DEFAULT_IMPLEMENTATION:
            model = self.implementations[implementation](model) or model
        if self.prepare_fn is not None:
            model = self.prepare_fn(model) or model
        return model


def spec_of(entry: str) -> BenchmarkSpec:
    """The ``BenchmarkSpec`` that a model_zoo entry declares."""
    from tests.kit.model_zoo import model_zoo

    if entry not in model_zoo:
        raise ValueError(f"no model_zoo entry {entry!r}")
    attribute = model_zoo[entry][4]
    spec = attribute.benchmark if attribute is not None else None
    if spec is None:
        raise ValueError(f"{entry} declares no ModelAttribute.benchmark")
    return spec

"""What a model's contributors declare once per model: ``BenchmarkSpec``, attached to a model_zoo entry through
``ModelAttribute.benchmark``.

model_zoo imports this module, so it must not import ColossalAI. An implementation that swaps in ColossalAI modules
(``ColoAttention``, the MoE layer) imports them inside its function, which runs only when that implementation is
measured.
"""

import importlib
import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

# the model exactly as ``BenchmarkSpec.models`` builds it
DEFAULT_IMPLEMENTATION = "default"
# what a model's weights are: its published checkpoint, or randomly initialized
PRETRAINED, RANDOM = "pretrained", "random"
WEIGHTS = (PRETRAINED, RANDOM)
# what a maximum search can change
SEARCHABLE = ("seq_len", "micro_batch")


@dataclass(frozen=True)
class HFModel:
    """A model published on the HuggingFace Hub, pinned to a revision, built either with its published weights or,
    from its configuration alone, with random ones.

    The repository is named, never a path: files are found through the standard HuggingFace variables
    (``HF_HUB_CACHE``, ``HF_HUB_OFFLINE``), e.g. a shared offline cache,
    elsewhere the Hub.

    Args:
        repo: Repository id, e.g. ``"Qwen/Qwen3-1.7B"``.
        revision: The commit of the repository the numbers are for; the configuration and weights of another commit
            may differ.
        model_class: Dotted path of the class, e.g. ``"transformers.Qwen3ForCausalLM"`` or
            ``"diffusers.WanTransformer3DModel"``. Imported only when a model is built.
        config: The configuration as keyword arguments, for a model whose configuration is written out here (so that
            building it with random weights needs no download); by default it is read from the repository.
        config_overrides: Settings on top of the configuration, e.g. ``{"attn_implementation": "sdpa"}``.
        subfolder: For a diffusers pipeline repository, the component (e.g. ``"transformer"``).
        trust_remote_code: The repository's own modeling code is run (remote-code models).
    """

    repo: str
    revision: str
    model_class: str
    config: Optional[Mapping[str, Any]] = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    subfolder: Optional[str] = None
    trust_remote_code: bool = False

    def _cls(self):
        module, name = self.model_class.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)

    def __call__(self, pretrained: bool = False) -> nn.Module:
        cls = self._cls()
        where = (
            dict(revision=self.revision, subfolder=self.subfolder) if self.subfolder else dict(revision=self.revision)
        )
        # offline: diffusers would otherwise ask the Hub for the list of files before loading
        from huggingface_hub import constants

        where["local_files_only"] = constants.HF_HUB_OFFLINE
        if self.model_class.startswith("diffusers."):
            if pretrained:
                return cls.from_pretrained(self.repo, torch_dtype=torch.float32, **where, **self.config_overrides)
            config = dict(self.config) if self.config is not None else cls.load_config(self.repo, **where)
            return cls.from_config({**config, **self.config_overrides})
        from transformers import AutoConfig

        remote = dict(trust_remote_code=self.trust_remote_code)
        if self.config is not None:
            config = cls.config_class(**self.config, **self.config_overrides)
        else:
            config = AutoConfig.from_pretrained(self.repo, **where, **remote, **self.config_overrides)
        if pretrained:
            return cls.from_pretrained(self.repo, config=config, torch_dtype=torch.float32, **where, **remote)
        if cls.__name__.startswith("Auto"):
            return cls.from_config(config, **remote)
        return cls(config)


def causal_lm_batch(model: nn.Module, micro_batch: int, seq_len: int, generator: torch.Generator) -> Dict[str, Tensor]:
    """Random token ids over the model's whole vocabulary, with the inputs as labels (HuggingFace shifts them)."""
    input_ids = torch.randint(model.config.vocab_size, (micro_batch, seq_len), generator=generator)
    return dict(input_ids=input_ids, labels=input_ids.clone())


def count_input_ids(batch: Dict[str, Tensor]) -> int:
    return batch["input_ids"].numel()


def model_loss(model: nn.Module, batch: Dict[str, Any]) -> Tensor:
    """``model(**batch).loss``: a HuggingFace model given its labels computes its own loss."""
    return model(**batch).loss


def enable_checkpointing(model: nn.Module, preserve_rng_state: bool = True) -> None:
    """Non-reentrant gradient checkpointing through the model's own switch: ``gradient_checkpointing_enable`` of a
    transformers model, ``enable_gradient_checkpointing`` of a diffusers model. A model with neither (e.g. remote code
    with its own switch) declares ``BenchmarkSpec.checkpointing_fn``."""
    kwargs = {"use_reentrant": False, "preserve_rng_state": preserve_rng_state}
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    elif hasattr(model, "enable_gradient_checkpointing"):
        if "gradient_checkpointing_func" in inspect.signature(model.enable_gradient_checkpointing).parameters:
            from torch.utils.checkpoint import checkpoint

            model.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, **kwargs))
        else:
            model.enable_gradient_checkpointing()
    else:
        raise NotImplementedError(
            f"{type(model).__name__} has no gradient_checkpointing_enable or enable_gradient_checkpointing; declare "
            "BenchmarkSpec.checkpointing_fn"
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    """How to benchmark a model_zoo entry at its real sizes.

    The entry's own ``model_fn`` builds a reduced model for unit tests; ``models`` builds the sizes that are actually
    trained. Each step feeds one batch of fixed shape, built once before the first step, so data loading is not part
    of the measurement.

    Args:
        models: Size name (e.g. ``"8b"``) to the model at that size, built on the CPU, in float32: an ``HFModel``,
            which has published weights (measured by default) and random ones, or a function with random weights
            only. The first size is the default.
        seq_len: Default sequence length.
        micro_batch: Default number of samples per device per step.
        batch_fn: ``batch_fn(model, micro_batch, seq_len, generator)`` returns one batch for ``model(**batch)``, on the
            CPU. Defaults to random token ids with themselves as labels (``causal_lm_batch``).
        tokens_fn: ``tokens_fn(batch)`` is the number of tokens a batch contributes to throughput. Defaults to the
            number of input ids; a diffusion model counts latent tokens.
        loss_fn: ``loss_fn(model, batch)`` runs the forward and returns the scalar the step backpropagates, so a loss
            that needs more than the model's output (a diffusion target, a mask) takes it from the batch. Defaults to
            ``model(**batch).loss``.
        prepare_fn: Called on every freshly built model, after its implementation is applied, e.g. to freeze a tower.
            It may modify the module in place and return ``None``, or return a replacement.
        checkpointing_fn: ``checkpointing_fn(model, preserve_rng_state)`` turns on non-reentrant gradient
            checkpointing. Defaults to the model's own switch (``enable_checkpointing``).
        flops_fn: ``flops_fn(model, seq_len)`` is the model FLOPs per token from a closed formula, given the model built
            on the meta device, e.g. ``metrics.compute.decoder_flops_per_token``. Required for a published report: MFU uses
            it, and the automatic count only checks it. Without one, MFU comes from the count and is flagged as not
            verified.
        flop_formulas: FLOPs of custom operators the automatic count cannot see otherwise, as
            ``torch.utils.flop_counter.FlopCounterMode``'s ``custom_mapping``: an operator (e.g.
            ``torch.ops.mylib.fused_attention``) to ``fn(*args, out_val=..., **kwargs)`` returning its FLOPs. For
            kernels registered as torch custom operators with a fake (meta) implementation.
        implementations: Other ways to compute the same model, by name, e.g. ``{"fused": use_fused_kernels}``. Each
            function takes the model as built (the ``"default"`` implementation) and swaps its modules, in place
            (returning ``None``) or by returning a replacement. Implementations compute the same FLOPs, so they share
            one FLOPs count.
        reference_implementation: The implementation whose FLOPs are counted. Counting runs the model on the meta
            device, which fused kernels (flash attention, a Triton kernel) and communicating layers (expert
            parallelism) cannot; the reference must be one that can, usually the plain PyTorch one.
        shardformer_policy: ``shardformer_policy()`` returns the shardformer ``Policy`` that a plugin applying one
            (HybridParallelPlugin) uses for this model; without it, the plugin picks ColossalAI's policy for the model
            class, if it has one.
        count_on_device: Count the FLOPs in one extra, untimed step of every measurement, on the devices, instead of
            on the meta device beforehand: for a model whose computation depends on values (a MoE router), which the
            meta device does not have.
        searchable: The fields a maximum search may change. A video model's sequence is set by its frames and
            resolution, so only its micro-batch is searched.
    """

    models: Mapping[str, Callable[[], nn.Module]]
    seq_len: int = 4096
    micro_batch: int = 1
    batch_fn: Callable[[nn.Module, int, int, torch.Generator], Dict[str, Any]] = causal_lm_batch
    tokens_fn: Callable[[Dict[str, Any]], int] = count_input_ids
    loss_fn: Callable[[nn.Module, Dict[str, Any]], Tensor] = model_loss
    prepare_fn: Optional[Callable[[nn.Module], Optional[nn.Module]]] = None
    checkpointing_fn: Callable[[nn.Module, bool], None] = enable_checkpointing
    flops_fn: Optional[Callable[[nn.Module, int], float]] = None
    flop_formulas: Mapping[Any, Callable[..., int]] = field(default_factory=dict)
    implementations: Mapping[str, Callable[[nn.Module], Optional[nn.Module]]] = field(default_factory=dict)
    reference_implementation: str = DEFAULT_IMPLEMENTATION
    shardformer_policy: Optional[Callable[[], Any]] = None
    count_on_device: bool = False
    searchable: Tuple[str, ...] = SEARCHABLE

    def __post_init__(self):
        if not self.models:
            raise ValueError("BenchmarkSpec.models needs at least one size")
        if DEFAULT_IMPLEMENTATION in self.implementations:
            raise ValueError(f"{DEFAULT_IMPLEMENTATION!r} is the model as built; name the other implementations")
        if not set(self.searchable) <= set(SEARCHABLE):
            raise ValueError(f"can search {SEARCHABLE}, not {self.searchable}")
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

    def has_pretrained(self, size: str) -> bool:
        return isinstance(self.models[size], HFModel)

    def build(self, size: str, implementation: str = DEFAULT_IMPLEMENTATION, pretrained: bool = False) -> nn.Module:
        """The model at ``size`` in ``implementation``, with its published or random weights, prepared; on the current
        default device."""
        if size not in self.models:
            raise ValueError(f"no size {size!r}; sizes: {list(self.models)}")
        if implementation not in self.implementation_names:
            raise ValueError(f"no implementation {implementation!r}; implementations: {self.implementation_names}")
        source = self.models[size]
        if pretrained and not self.has_pretrained(size):
            raise ValueError(f"size {size!r} has no published weights declared; measure it with random weights")
        model = source(pretrained=pretrained) if isinstance(source, HFModel) else source()
        if implementation != DEFAULT_IMPLEMENTATION:
            model = self.implementations[implementation](model) or model
        if self.prepare_fn is not None:
            model = self.prepare_fn(model) or model
        return model


def spec_of(entry: str) -> BenchmarkSpec:
    """The ``BenchmarkSpec`` that a model_zoo entry declares, or a benchmark-only entry (``entries.ENTRIES``) for a
    model that has no model_zoo entry yet."""
    from tests.kit.model_zoo import model_zoo

    from ..entries import ENTRIES

    if entry not in model_zoo:
        if entry in ENTRIES:
            return ENTRIES[entry]
        raise ValueError(f"no model_zoo entry {entry!r} and no benchmark entry; benchmark entries: {sorted(ENTRIES)}")
    attribute = model_zoo[entry][4]
    spec = attribute.benchmark if attribute is not None else None
    if spec is None:
        raise ValueError(f"{entry} declares no ModelAttribute.benchmark")
    return spec

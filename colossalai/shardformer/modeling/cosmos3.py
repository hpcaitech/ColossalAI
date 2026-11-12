r"""Cosmos 3 attention on ColossalAI's two-way flat-attention kernel.

Cosmos 3 keeps its attention core in a replaceable ``processor`` object, and its forward only calls
``self.processor(self, und_seq, gen_seq, rotary_emb)``. :class:`ColossalTwoWayAttnProcessor` keeps the stock
``Cosmos3AttnProcessor`` prologue -- per-pathway projections, q/k norms, 3D RoPE -- and hands the understanding (AR)
prefix and the generation (DM) suffix to
:func:`~colossalai.shardformer.layer.two_way_attn.two_way_flat_attention` in one call instead of two
``dispatch_attention_fn`` calls.

The per-sample boundaries of a packed batch travel beside the forward arguments as a
:class:`~colossalai.shardformer.layer.two_way_attn.TwoWayPackedMeta`; the layout is described on the processor class.
Only the attribute contract of the stock processor is used, so this module imports without diffusers.
"""

from typing import Optional, Tuple

import torch

from ..layer.two_way_attn import TwoWayPackedMeta, two_way_flat_attention

__all__ = [
    "ColossalTwoWayAttnProcessor",
    "Cosmos3TwoWayAttnProcessor",
    "set_cosmos3_packed_meta",
    "install_fp32_timestep_embedding",
]

_ATTENTION_CLS = "Cosmos3PackedMoTAttention"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half, matching the diffusers Cosmos 3 processor."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class ColossalTwoWayAttnProcessor:
    """Drop-in replacement for diffusers' ``Cosmos3AttnProcessor`` backed by the two-way kernel.

    A pack of ``N`` samples is laid out as a global AR block followed by a global DM block, sample ``i`` owning
    ``ar_lens[i]`` contiguous AR tokens and ``dm_lens[i]`` contiguous DM tokens. That is the split ``und_len`` already
    describes, so the per-sample boundaries are set on the processor -- one instance is shared by every decoder layer
    -- before the forward; without them the whole pack counts as one sample, the stock batch-size-one behaviour.

    Args:
        backend (str, optional): ``"naive"`` for the PyTorch reference or ``"npu"`` for the Ascend fusion path.
            Defaults to ``"naive"`` so the processor also runs on CPU and GPU.
        packed_meta (Optional[TwoWayPackedMeta], optional): Per-sample AR/DM lengths of the pack. When omitted, every
            forward is treated as a single sample whose lengths are read from the two input tensors.

    Raises:
        ValueError: If ``backend`` is unknown.
    """

    # Kept for drop-in parity with the stock processor: diffusers' `set_attention_backend` writes here. The two
    # attention calls are chosen by `backend` instead, so a value written here is recorded and otherwise unused.
    _attention_backend = None
    _parallel_config = None

    def __init__(self, backend: str = "naive", packed_meta: Optional[TwoWayPackedMeta] = None) -> None:
        if backend not in ("naive", "npu"):
            raise ValueError(f"backend must be 'naive' or 'npu', got {backend!r}.")
        self.backend = backend
        self.packed_meta: Optional[TwoWayPackedMeta] = None
        if packed_meta is not None:
            self.set_packed_meta(packed_meta)

    def set_packed_meta(self, packed_meta) -> None:
        """Set the current pack's per-sample boundaries.

        Args:
            packed_meta (Optional[TwoWayPackedMeta]): The metadata of the next pack, or a
                :class:`~colossalai.shardformer.modeling.cosmos_mot.CosmosTwoWayLayout` carrying one. ``None`` restores
                the single-sample fallback.

        Raises:
            TypeError: If the value is neither ``None``, a ``TwoWayPackedMeta`` nor an object with a ``meta`` field.
        """
        if packed_meta is not None and not isinstance(packed_meta, TwoWayPackedMeta):
            # Accept the packer-facing layout object as well, so callers do not have to reach into `.meta`.
            packed_meta = getattr(packed_meta, "meta", packed_meta)
        if packed_meta is not None and not isinstance(packed_meta, TwoWayPackedMeta):
            raise TypeError(
                "packed_meta must be a TwoWayPackedMeta, a CosmosTwoWayLayout or None, got "
                f"{type(packed_meta).__name__}."
            )
        self.packed_meta = packed_meta

    def _resolve_meta(self, num_ar_tokens: int, num_dm_tokens: int) -> TwoWayPackedMeta:
        if self.packed_meta is None:
            return TwoWayPackedMeta([num_ar_tokens], [num_dm_tokens])
        meta = self.packed_meta
        if meta.total_ar_tokens != num_ar_tokens or meta.total_dm_tokens != num_dm_tokens:
            raise ValueError(
                f"The packed metadata describes {meta.total_ar_tokens} AR and {meta.total_dm_tokens} DM tokens, but "
                f"the attention inputs carry {num_ar_tokens} AR and {num_dm_tokens} DM tokens. Set the metadata from "
                f"the same pack that produced the inputs."
            )
        return meta

    def __call__(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-pathway projections, exactly as the stock Cosmos3AttnProcessor.
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        # Cosmos Edge normalizes the AR keys once more for the generation pathway; other variants leave this None, in
        # which case the generation pass reuses the AR key block as-is.
        k_norm_und_for_gen = getattr(attn, "k_norm_und_for_gen", None)
        k_und_for_gen = None if k_norm_und_for_gen is None else k_norm_und_for_gen(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        # Apply rotary position embeddings per pathway.
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        if k_und_for_gen is not None:
            k_und_for_gen = k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen

        meta = self._resolve_meta(q_und.shape[0], q_gen.shape[0])
        und_out, gen_out = two_way_flat_attention(
            q_und,
            k_und,
            v_und,
            q_gen,
            k_gen,
            v_gen,
            meta,
            backend=self.backend,
            k_ar_for_gen=k_und_for_gen,
        )

        # Per-pathway output projection.
        return attn.to_out(und_out.flatten(-2, -1)), attn.to_add_out(gen_out.flatten(-2, -1))


# The model-prefixed spelling reads better from framework code; the historical name is kept for the SFT scripts.
Cosmos3TwoWayAttnProcessor = ColossalTwoWayAttnProcessor


def install_fp32_timestep_embedding(model: torch.nn.Module) -> int:
    """Run the timestep embedder's matmuls in fp32 while its parameters stay in the training dtype.

    The reference implementation of this model embeds timesteps in fp32 on purpose -- "Timesteps are computed in
    FP32 for numerical stability" -- by opening ``torch.autocast(dtype=torch.float32)`` around the call, while
    FSDP2 has already cast the module's parameters to bf16. Measured on its own training step: the parameters
    read ``bfloat16`` at the module's forward and the autocast region is open in ``float32``. So the arrangement
    being reproduced here is bf16 storage with fp32 arithmetic, not fp32 storage.

    autocast is not how it is reproduced. ``torch.autocast`` with ``dtype=torch.float32`` is a CUDA-shaped
    construct whose support on other accelerators is not something this policy should depend on, and an ambient
    autocast also reaches code it was not aimed at. Instead the module's forward is replaced by one that calls
    the original through ``torch.func.functional_call`` with fp32 copies of its parameters and buffers. That is
    arithmetically the same thing -- every matmul and activation inside runs in fp32 -- it reimplements none of
    the module's own logic, and the gradient reaches the real parameters through the ``.float()`` that produced
    the copies. The result is returned in the dtype the caller handed in, so nothing downstream changes shape or
    dtype.

    What this does NOT recover on its own: the model's forward computes the sinusoidal projection and then
    casts it to the embedder's parameter dtype *before* calling it -- ``self.time_embedder(self.time_proj(t)
    .to(dtype))`` -- so the input arrives already rounded, and upcasting it here cannot undo that.
    :func:`install_fp32_sinusoidal_timestep_input` closes that part, from the other side of the same call.

    Args:
        model (torch.nn.Module): The transformer. Every submodule named ``time_embedder`` is wrapped; a model
            without one is left alone and reported as zero.

    Returns:
        int: How many modules were wrapped, so a caller can fail instead of silently wrapping nothing.
    """
    wrapped = 0
    for name, module in model.named_modules():
        if not name.endswith("time_embedder") or getattr(module, "_colossal_fp32_forward", False):
            continue
        original = module.__class__.forward

        def forward(self, *args, _original=original, **kwargs):
            # ``functional_call`` invokes ``self(...)``, which lands back here; the flag turns the inner call
            # into the module's own forward, now reading the fp32 parameters that functional_call installed.
            if getattr(self, "_colossal_fp32_reentrant", False):
                return _original(self, *args, **kwargs)
            dtype = next((a.dtype for a in args if torch.is_tensor(a) and a.is_floating_point()), torch.float32)
            tensors = {
                key: value.float()
                for key, value in list(self.named_parameters()) + list(self.named_buffers())
                if value is not None and value.is_floating_point()
            }
            cast_args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
            self._colossal_fp32_reentrant = True
            try:
                out = torch.func.functional_call(self, tensors, cast_args, kwargs)
            finally:
                self._colossal_fp32_reentrant = False
            return out.to(dtype) if torch.is_tensor(out) else out

        module.forward = forward.__get__(module, module.__class__)
        module._colossal_fp32_forward = True
        wrapped += 1
    return wrapped


def install_fp32_sinusoidal_timestep_input(model: torch.nn.Module) -> int:
    """Hand the timestep embedder the sinusoidal projection as computed, before it is rounded.

    The model's forward is ``self.time_embedder(self.time_proj(timesteps).to(time_embedder_dtype))``. The
    projection itself runs in fp32, because the timesteps are fp32, and is then rounded to the embedder's
    parameter dtype -- bf16 under a mixed-precision plugin. The reference implementation has no such
    rounding: its ``TimestepEmbedder.forward`` builds the sinusoidal features and feeds them to its MLP
    inside one module, under an fp32 autocast region, and only the *output* is cast to the model dtype.

    The rounding happens in the model's own forward, between the two submodules, so replacing either of them
    cannot remove it. What can be done from outside is to keep the value before the cast: ``time_proj``'s
    forward stores its output, and ``time_embedder``'s forward uses that stored value in place of the rounded
    copy it was handed. The two are the same tensor up to the cast, and that is asserted on every call rather
    than assumed -- if the model ever stops calling them as a pair, this raises instead of quietly going back
    to the rounded input, which is the failure mode worth avoiding: it would be invisible in every output.

    Install this after :func:`install_fp32_timestep_embedding`, and only after it: an embedder whose
    parameters are still bf16 cannot be handed fp32 features at all (the matmul refuses mixed dtypes), so a
    pair without that wrapper raises here rather than at the first forward. The embedding leaves in the
    dtype the caller's rounded copy had, which is what the model dtype is and what the reference
    implementation also returns -- fp32 is for the arithmetic, not for what the rest of the model holds. Measured on one Edge step with the parameters in bf16: the
    features themselves move by relL2 7.0e-04 and the embedding by 1.7e-03, which is 0.026% on the vision
    loss but relL2 1.2e-02 on the vision prediction -- the same order as the gap to the reference.

    Args:
        model (torch.nn.Module): The transformer. Every module that owns both a ``time_proj`` and a
            ``time_embedder`` child is wrapped; a model without such a pair is left alone and reported as
            zero.

    Returns:
        int: How many module pairs were wrapped, so a caller can fail instead of silently wrapping nothing.
    """
    wrapped = 0
    for parent in model.modules():
        proj = getattr(parent, "time_proj", None)
        embedder = getattr(parent, "time_embedder", None)
        if proj is None or embedder is None or getattr(parent, "_colossal_fp32_sinusoid_installed", False):
            continue
        if not getattr(embedder, "_colossal_fp32_forward", False):
            raise RuntimeError(
                "install_fp32_sinusoidal_timestep_input needs install_fp32_timestep_embedding on the same "
                "embedder first: handing fp32 features to a module whose parameters are still in the "
                "training dtype fails in the first matmul."
            )
        stash = {}
        proj_forward, embedder_forward = proj.forward, embedder.forward

        def projected(*args, _inner=proj_forward, _stash=stash, **kwargs):
            out = _inner(*args, **kwargs)
            if torch.is_tensor(out):
                _stash["features"] = out
            return out

        def embedded(sample, *args, _inner=embedder_forward, _stash=stash, **kwargs):
            features = _stash.pop("features", None)
            if features is None or not torch.is_tensor(sample):
                return _inner(sample, *args, **kwargs)
            if features.shape != sample.shape or not torch.equal(sample, features.to(sample.dtype)):
                raise RuntimeError(
                    "the timestep embedder was not handed the sinusoidal projection this pass computed, so "
                    "the fp32 input cannot be restored. The model no longer calls time_proj and "
                    "time_embedder as a pair; install_fp32_sinusoidal_timestep_input has to be updated "
                    "rather than left to fall back on the rounded input."
                )
            out = _inner(features, *args, **kwargs)
            # Back to what the caller would have received. The rounded copy carries the model dtype, and
            # everything downstream of the embedding is in it.
            if torch.is_tensor(out) and out.is_floating_point():
                out = out.to(sample.dtype)
            return out

        proj.forward = projected
        embedder.forward = embedded
        parent._colossal_fp32_sinusoid_installed = True
        wrapped += 1
    return wrapped


def set_cosmos3_packed_meta(model: torch.nn.Module, packed_meta) -> int:
    """Set the pack metadata on every Cosmos 3 attention processor in ``model``.

    This is the plugin-agnostic path: it walks the module tree, so it also works through a ``Booster`` wrapper
    (``ModelWrapper`` registers the wrapped module as a child) and when the policy object is not at hand.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        packed_meta (Optional[TwoWayPackedMeta]): Metadata for the next forward; see
            :meth:`ColossalTwoWayAttnProcessor.set_packed_meta`.

    Returns:
        int: Number of attention processors updated.
    """
    updated = 0
    for module in model.modules():
        if module.__class__.__name__ != _ATTENTION_CLS:
            continue
        processor = getattr(module, "processor", None)
        setter = getattr(processor, "set_packed_meta", None)
        if setter is not None:
            setter(packed_meta)
            updated += 1
    return updated

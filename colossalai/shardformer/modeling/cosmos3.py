r"""Cosmos 3 MoT attention on ColossalAI's two-way flat-attention kernel.

The diffusers Cosmos 3 transformer keeps its attention core in a replaceable ``processor`` object:
``Cosmos3PackedMoTAttention.forward`` is only

.. code-block:: python

    return self.processor(self, und_seq, gen_seq, rotary_emb)

and the stock ``Cosmos3AttnProcessor`` does the per-pathway projections, q/k norms, 3D RoPE and the two attention
calls. :class:`ColossalTwoWayAttnProcessor` keeps that prologue and replaces the two attention calls with
:func:`~colossalai.shardformer.layer.two_way_attn.two_way_flat_attention`, so the understanding (AR) prefix and the
generation (DM) suffix are handed to one kernel instead of two ``dispatch_attention_fn`` calls.

The class deliberately does not import diffusers. It only relies on the attributes the stock processor reads
(``to_q``/``to_k``/``to_v``, ``add_q_proj``/``add_k_proj``/``add_v_proj``, ``to_out``/``to_add_out``, the q/k norms,
``num_attention_heads``, ``num_key_value_heads`` and ``head_dim``), which also makes it testable with a small stand-in
module and keeps ``colossalai.shardformer`` importable without the optional diffusers dependency.

Multi-sample packs
------------------
A pack of ``N`` samples is laid out as a global AR block followed by a global DM block: sample ``i`` owns
``ar_lens[i]`` contiguous AR tokens and ``dm_lens[i]`` contiguous DM tokens. The transformer's forward only has a
scalar ``und_len``, which is exactly the global AR/DM split point, so the per-sample boundaries have to travel
beside the forward arguments. They are carried by
:class:`~colossalai.shardformer.layer.two_way_attn.TwoWayPackedMeta` and set on the processor (one instance is shared
by every decoder layer, like diffusers' own ``enable_cosmos3_flash_attention`` helper) with
:meth:`ColossalTwoWayAttnProcessor.set_packed_meta` before the forward.

With no metadata the processor treats the whole pack as one sample, which is the original batch-size-one behaviour
and keeps single-sample smoke tests working without any plumbing.
"""

from typing import Optional, Tuple

import torch

from ..layer.two_way_attn import TwoWayPackedMeta, two_way_flat_attention

__all__ = [
    "ColossalTwoWayAttnProcessor",
    "Cosmos3TwoWayAttnProcessor",
    "set_cosmos3_packed_meta",
]

_ATTENTION_CLS = "Cosmos3PackedMoTAttention"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half, matching the diffusers Cosmos 3 processor."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class ColossalTwoWayAttnProcessor:
    """Drop-in replacement for diffusers' ``Cosmos3AttnProcessor`` backed by the two-way kernel.

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

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

from types import MethodType
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from ..layer._operation import gather_forward_reducescatter_backward, packed_all_to_all_single
from ..layer.two_way_attn import TwoWayPackedMeta, _load_cuda_varlen_attention, two_way_flat_attention

__all__ = [
    "ColossalTwoWayAttnProcessor",
    "ColossalTwoWayCPAttnProcessor",
    "Cosmos3ContextParallel",
    "Cosmos3TwoWayAttnProcessor",
    "enable_cosmos3_context_parallel",
    "enable_cosmos3_selective_checkpointing",
    "set_cosmos3_packed_meta",
    "clear_cosmos3_packed_meta",
    "install_fp32_timestep_embedding",
    "install_fp32_sinusoidal_timestep_input",
    "get_cosmos3_selective_checkpoint_stats",
]

_ATTENTION_CLS = "Cosmos3PackedMoTAttention"
_DECODER_LAYER_CLS = "Cosmos3VLTextMoTDecoderLayer"


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
        backend (str, optional): ``"naive"`` for the PyTorch reference, ``"npu"`` for the Ascend fusion path or
            ``"cuda"`` for PyTorch's native packed CUDA kernel. Defaults to ``"naive"`` so the processor also runs on
            CPU.
        packed_meta (Optional[TwoWayPackedMeta], optional): Per-sample AR/DM lengths of the pack. When omitted, every
            forward is treated as a single sample whose lengths are read from the two input tensors.
        scale (Optional[float], optional): Softmax scale forwarded to the two-way kernel. Defaults to None.
        dropout_p (float, optional): Attention dropout. The CUDA backend rejects a non-zero value because the native
            packed kernel has no dropout. Defaults to 0.0.

    Raises:
        ValueError: If ``backend`` is unknown.
    """

    # Diffusers writes this property when set_attention_backend is called. Only the actual two-way backend
    # names are supported: accepting arbitrary diffusers dispatch names here would silently ignore user intent.
    @property
    def _attention_backend(self):
        return self.backend

    @_attention_backend.setter
    def _attention_backend(self, value):
        if value not in ("naive", "cuda", "npu"):
            raise ValueError(
                f"Cosmos 3 two-way attention does not support diffusers backend {value!r}; "
                "choose 'naive', 'cuda' or 'npu' on the processor/policy."
            )
        if value == "cuda":
            _load_cuda_varlen_attention()
        self.backend = value

    _parallel_config = None

    def __init__(
        self,
        backend: str = "naive",
        packed_meta: Optional[TwoWayPackedMeta] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> None:
        if backend not in ("naive", "npu", "cuda"):
            raise ValueError(f"backend must be 'naive', 'npu' or 'cuda', got {backend!r}.")
        self.backend = backend
        self.scale = scale
        self.dropout_p = dropout_p
        # Warm the packed CUDA kernel here, not in the first forward: the loader imports through importlib, which
        # torch.compile refuses to trace, and the processor is built before the model is compiled.
        if backend == "cuda":
            _load_cuda_varlen_attention()
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

    def clear_packed_meta(self) -> None:
        """Drop the pack metadata, restoring the single-sample fallback for the next forward."""
        self.set_packed_meta(None)

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

    def _project(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Per-pathway q/k/v projections, q/k norms and 3D RoPE, exactly as the stock ``Cosmos3AttnProcessor``.

        Shared by the single-device and the context-parallel processors, which differ only in what they do with the
        returned tensors: a plain run hands them to the two-way kernel, a Ulysses run exchanges sequence for heads
        first. Everything here is token-local, so it gives the same answer on a sequence shard.

        Returns:
            tuple: ``(q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen)``, each of shape
            ``[tokens, heads, head_dim]``; ``k_und_for_gen`` is None unless the model normalizes the AR keys a second
            time for the generation pathway.
        """
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
        return q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen

    def __call__(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen = self._project(attn, und_seq, gen_seq, rotary_emb)

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
            scale=self.scale,
            dropout_p=self.dropout_p,
        )

        # Per-pathway output projection.
        return attn.to_out(und_out.flatten(-2, -1)), attn.to_add_out(gen_out.flatten(-2, -1))


# The model-prefixed spelling reads better from framework code; the historical name is kept for the SFT scripts.
Cosmos3TwoWayAttnProcessor = ColossalTwoWayAttnProcessor


class ColossalTwoWayCPAttnProcessor(ColossalTwoWayAttnProcessor):
    """Two-way attention under Ulysses context parallelism.

    Each rank starts with a contiguous slice of the packed sequence, so the projections, the q/k norms and the RoPE
    are already correct on it. Attention then needs the opposite layout: every query of the full sequence, for a
    fraction of the heads. Three all-to-alls per pathway (plus the optional AR-key-for-generation tensor) swap the two
    -- sequence gathered, heads scattered -- the two-way kernel runs on the full pack with the *global* per-sample
    metadata, and two all-to-alls restore the sequence-sharded layout. That is the Ulysses scheme the report uses
    (Sec. 5.2.3), and it keeps the two-way kernel unchanged: after the exchange each rank holds whole per-sample AR
    and DM blocks for its heads.

    Two details make it exact rather than approximate. The caller pads each pathway to a multiple of the context
    parallel size before sharding; the exchange therefore hands back padded blocks, which are trimmed to the real
    per-sample totals before the kernel and re-padded after it, so no padding token ever enters an attention score.
    And the per-sample boundaries are the ones the packer installed on the processor for the whole pack, not a
    re-derivation from the shard.

    Args:
        backend (str, optional): ``"naive"``, ``"npu"`` or ``"cuda"``, forwarded to the two-way kernel. Defaults to
            ``"naive"``.
        process_group (ProcessGroup, optional): The context parallel group. ``None`` is only valid on a single-rank
            world; the exchange is a no-op with a context parallel size of one. Defaults to None.
        packed_meta (Optional[TwoWayPackedMeta], optional): See :class:`ColossalTwoWayAttnProcessor`.
        scale (Optional[float], optional): Softmax scale forwarded to the two-way kernel. Defaults to None.
        dropout_p (float, optional): Attention dropout. Defaults to 0.0.
    """

    def __init__(
        self,
        backend: str = "naive",
        process_group=None,
        packed_meta: Optional[TwoWayPackedMeta] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> None:
        if process_group is None and dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError(
                "Cosmos 3 context parallelism needs an explicit process group when the world size is greater than "
                "1; pass the plugin's `sequence_parallel_process_group`."
            )
        super().__init__(backend=backend, packed_meta=packed_meta, scale=scale, dropout_p=dropout_p)
        self.process_group = process_group
        self.cp_size = 1 if process_group is None else dist.get_world_size(process_group)
        # Real (unpadded) lengths of the pack, stamped by the shard seam of the forward that is running.
        self.und_real: Optional[int] = None
        self.gen_real: Optional[int] = None

    def set_sequence_lengths(self, und_real: int, gen_real: int) -> None:
        """Record the real per-pathway totals of the pack whose shards the attention layers are about to see."""
        self.und_real, self.gen_real = int(und_real), int(gen_real)

    def _exchange(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the sequence and scatter the heads: ``[S_local, H, D]`` to ``[S_local * cp, H / cp, D]``."""
        if self.cp_size == 1:
            return x
        tokens, heads, head_dim = x.shape
        if tokens == 0:
            return x.new_empty((0, heads // self.cp_size, head_dim))
        if heads % self.cp_size != 0:
            raise ValueError(
                f"Cosmos3 attention heads ({heads}) must be divisible by the context parallel size ({self.cp_size})."
            )
        inner = heads // self.cp_size
        # Chunk j of the send buffer is destined for rank j, which keeps head chunk j.
        send = x.reshape(tokens, self.cp_size, inner, head_dim).permute(1, 0, 2, 3).contiguous()
        recv = packed_all_to_all_single(send.view(self.cp_size, -1), self.process_group)
        # Chunk i of the receive buffer comes from rank i, whose tokens precede this rank's in the pack, so the
        # sequence is already in pack order once the chunks are concatenated.
        return recv.view(self.cp_size, tokens, inner, head_dim).reshape(tokens * self.cp_size, inner, head_dim)

    def _exchange_back(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the heads and scatter the sequence: ``[S_padded, H / cp, D]`` to ``[S_padded / cp, H, D]``."""
        if self.cp_size == 1:
            return x
        tokens, inner, head_dim = x.shape
        if tokens == 0:
            return x.new_empty((0, inner * self.cp_size, head_dim))
        local = tokens // self.cp_size
        # Sequence chunk j is destined for rank j, which collects every rank's head chunk for it.
        send = x.reshape(self.cp_size, local, inner, head_dim).contiguous()
        recv = packed_all_to_all_single(send.view(self.cp_size, -1), self.process_group)
        return (
            recv.view(self.cp_size, local, inner, head_dim)
            .permute(1, 0, 2, 3)
            .reshape(local, inner * self.cp_size, head_dim)
        )

    def __call__(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.packed_meta
        if meta is None:
            raise RuntimeError(
                "Context-parallel Cosmos 3 attention needs the per-sample boundaries of the pack: call "
                "`policy.set_packed_meta(...)` from the packer before the forward."
            )
        if self.und_real is None or self.gen_real is None:
            raise RuntimeError(
                "Context-parallel Cosmos 3 attention needs the real sequence lengths recorded by the CP shard seam; "
                "run the model through `Cosmos3ContextParallel.shard` before the decoder layers."
            )
        if meta.total_ar_tokens != self.und_real or meta.total_dm_tokens != self.gen_real:
            raise ValueError(
                f"The packed metadata describes {meta.total_ar_tokens} AR and {meta.total_dm_tokens} DM tokens, but "
                f"the CP shard seam carries {self.und_real} AR and {self.gen_real} DM tokens. Set the metadata from "
                f"the same pack that produced the inputs."
            )
        expected_ar_tokens = (self.und_real + self.cp_size - 1) // self.cp_size
        expected_dm_tokens = (self.gen_real + self.cp_size - 1) // self.cp_size
        if und_seq.shape[0] != expected_ar_tokens or gen_seq.shape[0] != expected_dm_tokens:
            raise ValueError(
                f"The CP shards should carry {expected_ar_tokens} AR and {expected_dm_tokens} DM tokens for a context "
                f"parallel size of {self.cp_size}, but the attention inputs carry {und_seq.shape[0]} and "
                f"{gen_seq.shape[0]}."
            )
        q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen = self._project(attn, und_seq, gen_seq, rotary_emb)

        # Trim the padding the shard seam appended, so the kernel sees exactly the tokens the packer described.
        ar, dm = meta.total_ar_tokens, meta.total_dm_tokens
        q_und = self._exchange(q_und)[:ar]
        k_und = self._exchange(k_und)[:ar]
        v_und = self._exchange(v_und)[:ar]
        if k_und_for_gen is not None:
            k_und_for_gen = self._exchange(k_und_for_gen)[:ar]
        q_gen = self._exchange(q_gen)[:dm]
        k_gen = self._exchange(k_gen)[:dm]
        v_gen = self._exchange(v_gen)[:dm]

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
            scale=self.scale,
            dropout_p=self.dropout_p,
        )

        # Re-pad, so the second exchange moves the same number of tokens on every rank, then hand back this rank's
        # sequence slice.
        und_out = self._exchange_back(self._pad_output(und_out, und_seq.shape[0]))
        gen_out = self._exchange_back(self._pad_output(gen_out, gen_seq.shape[0]))
        return attn.to_out(und_out.flatten(-2, -1)), attn.to_add_out(gen_out.flatten(-2, -1))

    def _pad_output(self, x: torch.Tensor, local_tokens: int) -> torch.Tensor:
        padded = local_tokens * self.cp_size
        if x.shape[0] == padded:
            return x
        return torch.cat([x, x.new_zeros((padded - x.shape[0],) + tuple(x.shape[1:]))], dim=0)


class Cosmos3ContextParallel:
    """The two seams that put a ``Cosmos3OmniTransformer`` into Ulysses context parallelism.

    The model calls ``_cp_shard_fn`` once per forward, right after the embeddings and before the decoder stack, and
    ``_cp_gather_fn`` once before the output projections. The first pads each pathway to a multiple of the context
    parallel size and keeps this rank's contiguous slice; the second returns the full per-pathway output, trimmed to
    the real length. Everything between them -- norms, MLP, residuals -- is token-local and runs on the slice, while
    attention crosses the group through :class:`ColossalTwoWayCPAttnProcessor`.

    The pack metadata stays global: every rank runs the embedding stage on the whole pack and every rank decodes the
    whole pack, so the loss, the indexes and the per-sample bookkeeping are identical to a single-device run. Every
    rank in the group must build the same pack -- same samples, same noise and same timesteps -- because the shards
    are concatenated into one sequence; the seam checks lengths, metadata and rotary sizes. Use
    ``broadcast_pack`` to share completed inputs/targets, and the explicit ``validate_pack`` debug hash to check
    token/position/payload equality. Length checks alone do not establish that the packs are equal.

    Args:
        process_group (ProcessGroup, optional): The context parallel group. ``None`` is only valid on a single-rank
            world; a multi-rank world has to pass the group explicitly so a missing group cannot silently turn
            context parallelism into a no-op. Defaults to None.
        backend (str, optional): Two-way kernel backend. Defaults to ``"naive"``.
        packed_meta (Optional[TwoWayPackedMeta], optional): Pack metadata to install on the CP processor. Defaults to
            None, i.e. the caller sets it from the packer before the first forward.
        scale (Optional[float], optional): Softmax scale forwarded to the CP processor. Defaults to None.
        dropout_p (float, optional): Attention dropout forwarded to the CP processor. Defaults to 0.0.
    """

    def __init__(
        self,
        process_group=None,
        backend: str = "naive",
        packed_meta: Optional[TwoWayPackedMeta] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> None:
        if process_group is None and dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError(
                "Cosmos 3 context parallelism needs an explicit process group when the world size is greater than "
                "1; pass the plugin's `sequence_parallel_process_group`."
            )
        self.process_group = process_group
        self.cp_size = 1 if process_group is None else dist.get_world_size(process_group)
        self.cp_rank = 0 if process_group is None else dist.get_rank(process_group)
        self.processor = ColossalTwoWayCPAttnProcessor(
            backend=backend,
            process_group=process_group,
            packed_meta=packed_meta,
            scale=scale,
            dropout_p=dropout_p,
        )
        self._und_real: Optional[int] = None
        self._gen_real: Optional[int] = None

    @property
    def packed_meta(self) -> Optional[TwoWayPackedMeta]:
        """The pack metadata currently installed on the CP processor, or ``None``."""
        return self.processor.packed_meta

    def set_packed_meta(self, packed_meta) -> None:
        """Delegate the next pack's per-sample boundaries to the CP processor."""
        self.processor.set_packed_meta(packed_meta)

    def clear_packed_meta(self) -> None:
        """Delegate clearing the pack metadata to the CP processor."""
        self.processor.clear_packed_meta()

    def broadcast_pack(self, inputs=None, targets=None, meta=None, leader: int = 0, device=None):
        """Broadcast completed (inputs, targets, meta) from a group-local leader and install metadata.

        All CP ranks call outside compiled forward; only the leader supplies the completed pack. Metadata uses
        an object collective, tensor payloads use tensor broadcasts. ``device`` is the local destination device
        (CPU by default for gloo, current accelerator for NCCL/HCCL). Set the current accelerator device first.
        DP replicas call on separate CP groups and may sample independently. Inputs must be detached data.
        """
        import pickle

        from torch.utils._pytree import tree_flatten, tree_unflatten

        if not 0 <= leader < self.cp_size:
            raise ValueError("leader must be a rank within the CP group")
        if device is None:
            backend = str(dist.get_backend(self.process_group)) if self.cp_size > 1 else "gloo"
            device = "cuda" if backend == "nccl" else "npu" if backend == "hccl" else "cpu"
        device = torch.device(device)
        header, tensors = [None], []
        # Materialize all source buffers before the first collective, so a leader-side error is propagated.
        if self.cp_rank == leader:
            try:
                resolved = getattr(meta, "meta", meta)
                if not isinstance(resolved, TwoWayPackedMeta):
                    raise TypeError("broadcast_pack requires TwoWayPackedMeta")
                leaves, spec = tree_flatten((inputs, targets, resolved))
                description = []
                for value in leaves:
                    if isinstance(value, torch.Tensor):
                        tensor = value.detach().to(device).contiguous().clone()
                        tensors.append(tensor)
                        description.append((True, (tuple(tensor.shape), tensor.dtype)))
                    else:
                        description.append((False, value))
                header[0] = (None, spec, description)
                # Fail on unpicklable user metadata before peers wait for its serialized header.
                pickle.dumps(header[0])
            except Exception as error:
                header[0] = (str(error), None, None)
        source = dist.get_global_rank(self.process_group, leader) if self.cp_size > 1 else 0
        if self.cp_size > 1:
            dist.broadcast_object_list(header, src=source, group=self.process_group)
        error, spec, description = header[0]
        if error is not None:
            raise ValueError(f"CP leader could not prepare pack: {error}")
        leaves, tensor_index = [], 0
        allocation_error = None
        try:
            for is_tensor, value in description:
                if is_tensor:
                    tensor = (
                        tensors[tensor_index]
                        if self.cp_rank == leader
                        else torch.empty(value[0], dtype=value[1], device=device)
                    )
                    tensor_index += 1
                    leaves.append(tensor)
                else:
                    leaves.append(value)
        except Exception as error:
            allocation_error = str(error)
        errors = [allocation_error]
        if self.cp_size > 1:
            errors = [None] * self.cp_size
            dist.all_gather_object(errors, allocation_error, group=self.process_group)
        if any(error is not None for error in errors):
            raise RuntimeError(f"CP pack allocation failed: {errors}")
        for (is_tensor, _), value in zip(description, leaves):
            if is_tensor and self.cp_size > 1 and value.numel():
                dist.broadcast(value, src=source, group=self.process_group)
        pack = tree_unflatten(leaves, spec)
        self.set_packed_meta(pack[2])
        return pack

    def validate_pack(self, tokens=None, positions=None, payload=None) -> None:
        """Explicit debug hash check for nested tokens, global fp32 positions, payload and installed metadata.

        All ranks must call with the same categories, outside compiled forward. Includes shape/dtype and exact
        tensor bytes (CPU synchronization is deliberate). Local serialization failures are exchanged before any
        rank raises, as are mismatches, so peers cannot enter attention while one rank exits this check.
        """
        import hashlib
        import pickle

        def canonical(value):
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                return (
                    str(tensor.dtype),
                    tuple(tensor.shape),
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
                )
            if isinstance(value, TwoWayPackedMeta):
                return (value.ar_lens, value.dm_lens)
            if isinstance(value, dict):
                return [(key, canonical(value[key])) for key in sorted(value)]
            if isinstance(value, (tuple, list)):
                return [canonical(item) for item in value]
            return value

        try:
            digest = hashlib.sha256(pickle.dumps(canonical((tokens, positions, payload, self.packed_meta)))).hexdigest()
            local = (None, digest)
        except Exception as error:
            local = (str(error), None)
        gathered = [local]
        if self.cp_size > 1:
            gathered = [None] * self.cp_size
            dist.all_gather_object(gathered, local, group=self.process_group)
        if any(item[0] is not None for item in gathered) or len({item[1] for item in gathered}) != 1:
            raise ValueError(f"CP pack debug hash mismatch or serialization error: {gathered}")

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.shape[0] % self.cp_size
        if remainder == 0:
            return x
        return torch.cat([x, x.new_zeros((self.cp_size - remainder,) + tuple(x.shape[1:]))], dim=0)

    def _shard(self, x: torch.Tensor) -> torch.Tensor:
        return torch.chunk(self._pad(x), self.cp_size, dim=0)[self.cp_rank]

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        if self.cp_size == 1:
            return x
        return gather_forward_reducescatter_backward(x, self.process_group, dim=0)

    def shard(self, und_seq, gen_seq, rotary_emb):
        """The ``_cp_shard_fn`` seam: pad each pathway, keep this rank's contiguous token slice."""
        self._und_real, self._gen_real = int(und_seq.shape[0]), int(gen_seq.shape[0])
        if self.cp_size > 1:
            # Context parallelism splits one pack, so every rank has to see the same real lengths. Checking them
            # here costs one tiny all-gather per forward and turns a silently mismatched pack into an error before
            # any attention collective can hang.
            lengths = torch.tensor(
                [self._und_real, self._gen_real],
                device=und_seq.device,
                dtype=torch.long,
            )
            gathered = [torch.empty_like(lengths) for _ in range(self.cp_size)]
            dist.all_gather(gathered, lengths, group=self.process_group)
            if any(tuple(item.tolist()) != (self._und_real, self._gen_real) for item in gathered):
                raise ValueError(
                    "Every rank in the context parallel group must see the same packed sequence; got "
                    f"{(self._und_real, self._gen_real)} on this rank and "
                    f"{[tuple(item.tolist()) for item in gathered]} across the group."
                )
        meta = self.packed_meta
        problem = None
        if meta is None or (meta.total_ar_tokens, meta.total_dm_tokens) != (
            self._und_real,
            self._gen_real,
        ):
            problem = "missing metadata or metadata totals differ from CP shard lengths"
        expected = (self._und_real, self._und_real, self._gen_real, self._gen_real)
        if len(rotary_emb) != 4 or any(
            not isinstance(t, torch.Tensor) or t.ndim == 0 or t.shape[0] != length
            for t, length in zip(rotary_emb, expected)
        ):
            problem = "rotary tensors must use global sequence positions before CP sharding"
        status = (problem, None if meta is None else (meta.ar_lens, meta.dm_lens))
        statuses = [status]
        if self.cp_size > 1:
            statuses = [None] * self.cp_size
            dist.all_gather_object(statuses, status, group=self.process_group)
        if any(item[0] is not None for item in statuses) or any(item[1] != status[1] for item in statuses):
            raise ValueError(f"CP pack metadata/rotary contract mismatch: {statuses}")
        self.processor.set_sequence_lengths(self._und_real, self._gen_real)
        return (
            self._shard(und_seq),
            self._shard(gen_seq),
            tuple(self._shard(tensor) for tensor in rotary_emb),
        )

    def gather(self, und_out, gen_out):
        """Reassemble full outputs; backward SUMs identical losses across CP ranks.

        Every CP rank computes the same full-pack loss. Average parameter gradients across CP (as well as
        independent DP replicas) exactly once to recover the single-pack gradient. A DP-only gradient reducer
        needs an explicit CP SUM/cp_size; do not also divide the loss by cp_size when averaging over DP*CP.
        """
        return (
            self._gather(und_out)[: self._und_real],
            self._gather(gen_out)[: self._gen_real],
        )


def enable_cosmos3_context_parallel(
    model: torch.nn.Module,
    process_group=None,
    backend: str = "naive",
    packed_meta: Optional[TwoWayPackedMeta] = None,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> Cosmos3ContextParallel:
    """Install Ulysses context parallelism on a Cosmos 3 omni transformer.

    Sets the context-parallel attention processor on every attention module, and wires the model's two CP seams to a
    :class:`Cosmos3ContextParallel` instance, which the caller keeps in order to call ``set_packed_meta`` on it.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer.
        process_group (ProcessGroup, optional): The context parallel group. ``None`` is only valid on a single-rank
            world; a multi-rank world has to pass the group explicitly. Defaults to None.
        backend (str, optional): Two-way kernel backend. Defaults to ``"naive"``.
        packed_meta (Optional[TwoWayPackedMeta], optional): Pack metadata to install on the CP processor. Defaults to
            None, i.e. the caller sets it from the packer before the first forward.
        scale (Optional[float], optional): Softmax scale forwarded to the CP processor. Defaults to None.
        dropout_p (float, optional): Attention dropout forwarded to the CP processor. Defaults to 0.0.

    Returns:
        Cosmos3ContextParallel: The helper owning the processor and the seams.

    Raises:
        RuntimeError: If the model has no CP seams, i.e. the installed diffusers predates them; silently setting the
            attributes would leave every rank with the full sequence and no error.
        ValueError: If ``process_group`` is None on a world with more than one rank.
    """
    if not hasattr(model, "_cp_shard_fn") or not hasattr(model, "_cp_gather_fn"):
        raise RuntimeError(
            f"{type(model).__name__} has no Cosmos 3 context-parallel seams (`_cp_shard_fn`/`_cp_gather_fn`); "
            "context parallelism needs a diffusers build that exposes them."
        )
    context_parallel = Cosmos3ContextParallel(
        process_group=process_group,
        backend=backend,
        packed_meta=packed_meta,
        scale=scale,
        dropout_p=dropout_p,
    )
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != _ATTENTION_CLS:
            continue
        module.set_processor(context_parallel.processor)
        installed += 1
    if installed == 0:
        raise RuntimeError(
            f"No {_ATTENTION_CLS} module found in {type(model).__name__}; context parallelism only applies to a "
            "Cosmos 3 omni transformer."
        )
    model._colossal_context_parallel = context_parallel
    model._cp_shard_fn = context_parallel.shard
    model._cp_gather_fn = context_parallel.gather
    return context_parallel


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
            dtype = next(
                (a.dtype for a in args if torch.is_tensor(a) and a.is_floating_point()),
                torch.float32,
            )
            tensors = {
                key: value.float()
                for key, value in list(self.named_parameters()) + list(self.named_buffers())
                if value is not None and value.is_floating_point()
            }
            cast_args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
            self._colossal_fp32_reentrant = True
            try:
                with torch.autocast(device_type=cast_args[0].device.type, enabled=False):
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
            if features.shape != sample.shape:
                raise RuntimeError(
                    "the timestep embedder was not handed the sinusoidal projection this pass computed, so "
                    "the fp32 input cannot be restored. The model no longer calls time_proj and "
                    "time_embedder as a pair; install_fp32_sinusoidal_timestep_input has to be updated "
                    "rather than left to fall back on the rounded input."
                )
            torch._assert_async(
                torch.all(sample == features.to(sample.dtype)),
                "timestep embedder was not handed the sinusoidal projection",
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


def set_cosmos3_packed_meta(model: torch.nn.Module, packed_meta, strict: bool = False) -> int:
    """Set the pack metadata on every Cosmos 3 attention processor in ``model``.

    This is the plugin-agnostic path: it walks the module tree, so it also works through a ``Booster`` wrapper
    (``ModelWrapper`` registers the wrapped module as a child) and when the policy object is not at hand.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        packed_meta (Optional[TwoWayPackedMeta]): Metadata for the next forward; see
            :meth:`ColossalTwoWayAttnProcessor.set_packed_meta`.
        strict (bool, optional): Raise if no Cosmos 3 attention processor was found. A training entry point should
            pass ``strict=True`` so a model that was not sharded through the policy fails loudly instead of training
            with the stock single-sample processor. Defaults to False.

    Returns:
        int: Number of attention processors updated.

    Raises:
        RuntimeError: If ``strict`` is true and no processor was updated.
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
    if strict and updated == 0:
        raise RuntimeError(
            f"No Cosmos 3 attention processor was found in {type(model).__name__}. Boost the model through "
            f"Cosmos3OmniPolicy before setting pack metadata, or call this helper on the unwrapped transformer."
        )
    return updated


def clear_cosmos3_packed_meta(model: torch.nn.Module, strict: bool = False) -> int:
    """Remove pack metadata from every Cosmos 3 attention processor in ``model``.

    Call this after a packed step has completed. It restores the stock single-sample behaviour, so a later forward
    cannot silently reuse the previous pack's boundaries.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        strict (bool, optional): Raise if no Cosmos 3 attention processor was found. Defaults to False.

    Returns:
        int: Number of attention processors updated.
    """
    return set_cosmos3_packed_meta(model, None, strict=strict)


def _checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb):
    """One Cosmos 3 decoder layer with its two MLPs under activation checkpointing.

    Same statements as the stock forward, in the same order: the input norms, the attention call, the two residuals.
    Only the checkpoint boundary moves -- around the MLP pair instead of around the whole layer -- so the backward
    pass recomputes the two up-projections and reuses the attention output the forward pass kept.
    """
    und_norm = self.input_layernorm(und_seq)
    gen_norm = self.input_layernorm_moe_gen(gen_seq)
    und_attn, gen_attn = self.self_attn(und_norm, gen_norm, rotary_emb)
    residual_und = und_seq + und_attn
    residual_gen = gen_seq + gen_attn

    def run_mlp(residual_und: torch.Tensor, residual_gen: torch.Tensor):
        mlp_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return mlp_und, mlp_gen

    # `use_reentrant=False` because the recomputation has to see the autograd graph the forward built, which is what
    # lets this sit inside a layer that is itself already running under an outer backward pass.
    mlp_und, mlp_gen = checkpoint(run_mlp, residual_und, residual_gen, use_reentrant=False)
    return residual_und + mlp_und, residual_gen + mlp_gen


# Dispatched op names that carry the attention output for the backends this policy runs on: PyTorch's native packed
# CUDA kernel, Ascend's fused attention, and the SDPA reference used by the CPU test. Anything else is recomputed.
_ATTENTION_OP_PATTERNS = (
    "varlen_attn",
    "fmha",
    "npu_fusion_attention",
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
)


def _is_attention_op(func) -> bool:
    """Whether a selective-checkpoint policy op is one whose output is worth keeping."""
    name = getattr(func, "__name__", "")
    return any(pattern in name for pattern in _ATTENTION_OP_PATTERNS)


class _SelectiveBudget:
    """Shape-dependent heuristic, not a hard memory cap (allocator/workspace/autograd overhead excluded)."""

    def __init__(self, budget_gib: float, num_layers: int) -> None:
        import math

        if not math.isfinite(budget_gib) or budget_gib < 0:
            raise ValueError("selective budget_gib must be finite and nonnegative")
        self.budget_bytes = float(budget_gib) * 2**30
        self.num_layers = int(num_layers)
        self.path = None
        self.max_saved_layers = 0
        self.signature = None
        self.stats = {}

    def decide(self, und_seq: torch.Tensor, gen_seq: torch.Tensor) -> None:
        signature = (
            tuple(und_seq.shape),
            tuple(gen_seq.shape),
            und_seq.dtype,
            gen_seq.dtype,
        )
        if signature == self.signature:
            return
        self.signature = signature
        tokens = int(und_seq.shape[0]) + int(gen_seq.shape[0])
        residual_bytes = max(
            und_seq.numel() * und_seq.element_size() + gen_seq.numel() * gen_seq.element_size(),
            1,
        )
        mlp_only_bytes = self.num_layers * residual_bytes * 3
        self.path = "mlp" if mlp_only_bytes <= self.budget_bytes else "sac"
        self.max_saved_layers = (
            self.num_layers if self.path == "mlp" else min(self.num_layers, int(self.budget_bytes / residual_bytes))
        )
        self.stats = {
            "mode": self.path,
            "tokens": tokens,
            "budget_bytes": self.budget_bytes,
            "estimated_mlp_bytes": mlp_only_bytes,
            "estimated_layer_output_bytes": residual_bytes,
            "estimated_saved_bytes": (mlp_only_bytes if self.path == "mlp" else self.max_saved_layers * residual_bytes),
            "max_saved_layers": self.max_saved_layers,
            "hard_budget": False,
            "attention_op_hits": {},
            "attention_op_saves": {},
            "save_count": 0,
        }


def get_cosmos3_selective_checkpoint_stats(model: torch.nn.Module) -> dict:
    """Return a detached logging snapshot; op names/counts are observed forward dispatches, not predictions.

    Counts accumulate for the current shape signature; a changed pack shape/dtype re-estimates and resets counts.
    Math SDPA may decompose into matmul/softmax, giving zero attention-op hits and saves on that backend.
    """
    import copy

    for module in model.modules():
        budget = getattr(module, "_colossal_selective_budget", None)
        if budget is not None:
            return copy.deepcopy(budget.stats)
    return {}


def _sac_checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb, budget: _SelectiveBudget):
    """Official-style SAC: checkpoint the whole layer and keep only the attention op outputs within budget."""
    layer_index = self._colossal_layer_index
    original_forward = self._colossal_original_forward
    max_saved_layers = budget.max_saved_layers
    stats = budget.stats

    def run_layer(und, gen, cos_und, sin_und, cos_gen, sin_gen):
        return original_forward(und, gen, (cos_und, sin_und, cos_gen, sin_gen))

    def policy(ctx, func, *args, **kwargs):
        # The decision is per layer, not per call: both the causal AR and the full DM attention calls in a selected
        # layer are kept, and the remaining layers recompute both.
        attention = _is_attention_op(func)
        save = layer_index < max_saved_layers and attention
        if attention and not ctx.is_recompute:
            name = str(func)
            stats["attention_op_hits"][name] = stats["attention_op_hits"].get(name, 0) + 1
            if save:
                stats["attention_op_saves"][name] = stats["attention_op_saves"].get(name, 0) + 1
                stats["save_count"] += 1
        if save:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return checkpoint(
        run_layer,
        und_seq,
        gen_seq,
        *rotary_emb,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
    )


def _selective_decoder_forward(self, und_seq, gen_seq, rotary_emb):
    """Dispatch to the MLP-only path for small packs and to budgeted SAC for large ones."""
    owner = self._colossal_checkpoint_owner
    if any(getattr(module, "gradient_checkpointing", False) for module in owner.modules()):
        raise ValueError("full gradient_checkpointing and selective checkpointing cannot be combined")
    if not self.training or not torch.is_grad_enabled():
        return self._colossal_original_forward(und_seq, gen_seq, rotary_emb)
    budget = self._colossal_selective_budget
    budget.decide(und_seq, gen_seq)
    if budget.path == "mlp":
        return _checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb)
    return _sac_checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb, budget)


def enable_cosmos3_selective_checkpointing(model: torch.nn.Module, budget_gib: float = 8.0) -> int:
    """Checkpoint Cosmos 3 decoder layers, adapting the saved set to the packed batch and a memory budget.

    Small packs keep the original MLP-only path: the attention output is kept and only the two MLP blocks are
    recomputed, which is the cheaper backward. When that path's estimated activation footprint would exceed
    ``budget_gib``, the layers switch to the official-style SAC path: each layer is checkpointed whole and only the
    attention op outputs of the first ``budget_gib / per-layer-output`` layers are kept, so the remaining layers
    recompute attention too. This reduces retained activations, but actual peak memory also depends on
    the backend workspaces, autograd state and allocator; it does not guarantee that a pack fits.

    The decision is re-estimated whenever the local pathway shapes or dtypes change. Estimates are heuristics,
    not a hard memory budget; attention dispatch/save counts are available through the statistics helper.
    ``budget_gib=0`` forces the SAC path with nothing saved, i.e. full recompute.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        budget_gib (float, optional): Memory budget for saved attention outputs (and, on the MLP-only path, the
            attention graph). Defaults to 8.0 GiB.

    Returns:
        int: Number of decoder layers patched; zero when they all were already.

    Raises:
        RuntimeError: If the model has no Cosmos 3 decoder layer, i.e. the patch would silently disable
            checkpointing and the step would run out of memory instead.
    """
    if any(getattr(module, "gradient_checkpointing", False) for module in model.modules()):
        raise ValueError("full gradient_checkpointing and selective checkpointing cannot be combined")
    layers = [module for module in model.modules() if module.__class__.__name__ == _DECODER_LAYER_CLS]
    if not layers:
        raise RuntimeError(
            f"No {_DECODER_LAYER_CLS} module found in {type(model).__name__}; selective checkpointing only applies "
            "to a Cosmos 3 omni transformer."
        )

    budget = _SelectiveBudget(budget_gib, len(layers))
    patched = 0
    for layer_index, module in enumerate(layers):
        if getattr(module, "_colossal_selective_checkpointing", False):
            continue
        # Avoid registering the root as a child (which would create a module traversal cycle).
        object.__setattr__(module, "_colossal_checkpoint_owner", model)
        module._colossal_original_forward = module.forward
        module._colossal_selective_budget = budget
        module._colossal_layer_index = layer_index
        module.forward = MethodType(_selective_decoder_forward, module)
        module._colossal_selective_checkpointing = True
        patched += 1
    return patched

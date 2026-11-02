"""Flash attention on Ascend NPUs through ``torch_npu.npu_fusion_attention``.

The kernel accepts the dense ``BNSD``/``BSND``/``BSH`` layouts and the packed ``TND`` layout. Dense tensors carry a
visibility mask; packed tensors are delimited by ``cu_seqlens_*`` instead.

Causal attention always passes a mask. The Ascend kernel ignores ``sparse_mode`` when ``atten_mask`` is ``None`` and
silently computes full attention, so this backend supplies the compressed ``[2048, 2048]`` lower-triangular mask that
``sparse_mode=3`` expects under the CANN 9.1 / torch_npu 2.13 contract, and caches it per device. A padded dense batch
from ``ColoAttention.prepare_attn_kwargs`` is un-padded by ``q_indices``/``kv_indices`` and served by the packed path.
"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch

from colossalai.kernel.attention_layout import QKV_LAYOUTS, AttentionLayout, attention_layout

from ...base_extension import _Extension

# CANN compresses causal masks for ``sparse_mode`` 2/3/4 and rejects any other last-two-dimension size with
# "please set the atten_mask_shape to [2048,2048]", so this is the only mask size ``sparse_mode=3`` accepts.
CAUSAL_MASK_SIZE = 2048

# ``sparse_mode`` of ``npu_fusion_attention``: 0 reads the mask it is given, 3 reads the compressed causal window.
_DENSE_MASK_MODE = 0
_CAUSAL_WINDOW_MODE = 3

__all__ = ["CAUSAL_MASK_SIZE", "FlashAttentionNpuExtension", "QKV_LAYOUTS", "resolve_fusion_attention_args"]


@dataclass(frozen=True)
class _PackedLengths:
    """Prefix sum ``[0, l0, l0 + l1, ...]`` of one side of a packed batch."""

    prefix: Tuple[int, ...]

    @property
    def lengths(self) -> Tuple[int, ...]:
        return tuple(end - start for start, end in zip(self.prefix, self.prefix[1:]))

    @property
    def actual(self) -> List[int]:
        """Cumulative ends, i.e. the prefix sum without its leading zero."""
        return list(self.prefix[1:])

    @property
    def longest(self) -> int:
        return max(self.lengths)

    def __len__(self) -> int:
        return len(self.prefix) - 1


@dataclass(frozen=True)
class _FusionAttentionRequest:
    """One attention call, before the backend-specific arguments are resolved."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dropout_p: float = 0.0
    scale: Optional[float] = None
    attention_mask: Optional[torch.Tensor] = None
    is_causal: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_kv: Optional[torch.Tensor] = None
    qkv_layout: str = "BNSD"
    head_num: Optional[int] = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_kv: Optional[int] = None


@dataclass(frozen=True)
class _FusionAttentionArgs:
    """The resolved arguments of ``npu_fusion_attention``."""

    head_num: int
    input_layout: str
    atten_mask: Optional[torch.Tensor]
    scale: float
    keep_prob: float
    actual_seq_qlen: Optional[List[int]]
    actual_seq_kvlen: Optional[List[int]]
    sparse_mode: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "head_num": self.head_num,
            "input_layout": self.input_layout,
            "atten_mask": self.atten_mask,
            "scale": self.scale,
            "keep_prob": self.keep_prob,
            "actual_seq_qlen": self.actual_seq_qlen,
            "actual_seq_kvlen": self.actual_seq_kvlen,
            "sparse_mode": self.sparse_mode,
        }

    def kernel_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "atten_mask": self.atten_mask,
            "scale": self.scale,
            "keep_prob": self.keep_prob,
            "sparse_mode": self.sparse_mode,
        }
        if self.actual_seq_qlen is not None:
            kwargs["actual_seq_qlen"] = self.actual_seq_qlen
            kwargs["actual_seq_kvlen"] = self.actual_seq_kvlen
        return {name: value for name, value in kwargs.items() if value is not None}


def _require_head_num(head_num: Optional[int]) -> int:
    if head_num is None:
        raise ValueError(
            "The BSH layout fuses the heads into the last dimension, so head_num is required to split it; pass "
            "head_num explicitly."
        )
    if head_num <= 0:
        raise ValueError(f"head_num must be positive, got {head_num}.")
    return head_num


def _read_cu_seqlens(cu_seqlens, argument_name: str, total_tokens: int) -> _PackedLengths:
    """Read and validate a cumulative-length prefix sum, touching the device at most once."""
    lengths = torch.as_tensor(cu_seqlens)
    if lengths.dim() != 1 or lengths.numel() < 2:
        raise ValueError(
            f"{argument_name} must be a 1D prefix sum [0, l0, l0 + l1, ...] describing at least one sequence, got "
            f"{lengths.tolist()} with shape {tuple(lengths.shape)}."
        )
    if lengths.dtype is torch.bool or lengths.dtype.is_floating_point or lengths.dtype.is_complex:
        raise TypeError(
            f"{argument_name} must contain integers, got dtype {lengths.dtype}. Cast the lengths with "
            f".to(torch.int32) before calling."
        )

    prefix = tuple(int(length) for length in lengths.tolist())
    if prefix[0] != 0:
        raise ValueError(
            f"{argument_name} must start at 0, got {list(prefix)}. Pass the cumulative lengths of the packed "
            f"sequence, i.e. [0, l0, l0 + l1, ...]."
        )
    if any(end < start for start, end in zip(prefix, prefix[1:])):
        raise ValueError(
            f"{argument_name} must be non-decreasing, got {list(prefix)}. Every sequence length must be "
            f"non-negative; a zero-length sequence is allowed."
        )
    if prefix[-1] != total_tokens:
        raise ValueError(
            f"{argument_name} must end at the number of tokens of the packed tensor, i.e. {total_tokens}, got "
            f"{prefix[-1]} ({list(prefix)})."
        )
    return _PackedLengths(prefix)


def _check_max_seqlen(value: Optional[int], lengths: _PackedLengths, argument_name: str) -> None:
    """The compressed mask no longer needs the maxima, but a wrong one still signals a caller bug."""
    if value is not None and int(value) < lengths.longest:
        raise ValueError(
            f"{argument_name}={value} is smaller than the longest sequence in the cumulative lengths "
            f"({lengths.longest}). Pass the true maximum or None."
        )


@lru_cache(maxsize=None)
def _causal_mask(device: torch.device) -> torch.Tensor:
    """Return the cached compressed causal mask, ``True`` where a position may not be attended.

    ``sparse_mode=3`` anchors the window at the bottom-right, so one mask serves every sequence length. It is
    cached per device because every layer of a training step reuses it.
    """
    return torch.ones(CAUSAL_MASK_SIZE, CAUSAL_MASK_SIZE, dtype=torch.bool, device=device).triu_(diagonal=1)


def _validate_qkv(request: _FusionAttentionRequest, layout: AttentionLayout) -> Tuple[int, int]:
    """Check that q, k and v agree, and return the head count and head dimension."""
    q, k, v = request.q, request.k, request.v
    if q.dim() != layout.rank or k.dim() != layout.rank or v.dim() != layout.rank:
        raise ValueError(
            f"q, k and v must be {layout.rank}D tensors in {layout.name} layout, got shapes "
            f"{tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
        )
    if layout.is_dense and not (q.size(0) == k.size(0) == v.size(0)):
        raise ValueError(
            f"q, k and v must have the same batch size in {layout.name} layout, got shapes "
            f"{tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have the same shape in {layout.name} layout, got {tuple(k.shape)} and {tuple(v.shape)}."
        )

    if layout.has_fused_heads:
        head_num = _require_head_num(request.head_num)
        if q.size(-1) != k.size(-1):
            raise ValueError(
                f"q, k and v must have the same fused head dimension in {layout.name} layout, got shapes "
                f"{tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
            )
        if q.size(-1) % head_num:
            raise ValueError(
                f"The last dimension of a {layout.name} tensor is heads * head_dim, so it must be divisible by "
                f"head_num, got {q.size(-1)} and {head_num}."
            )
        return head_num, q.size(-1) // head_num

    head_dim_index = layout.head_dim_index
    if q.size(head_dim_index) != k.size(head_dim_index):
        raise ValueError(
            f"q, k and v must have the same number of heads in {layout.name} layout (dimension {head_dim_index}), "
            f"got shapes {tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
        )
    if q.size(-1) != k.size(-1):
        raise ValueError(
            f"q, k and v must have the same head_dim in {layout.name} layout, got shapes {tuple(q.shape)}, "
            f"{tuple(k.shape)} and {tuple(v.shape)}."
        )
    head_num = q.size(head_dim_index)
    if request.head_num is not None and request.head_num != head_num:
        raise ValueError(
            f"head_num={request.head_num} disagrees with the {layout.name} tensors, which have {head_num} heads."
        )
    return head_num, q.size(-1)


def _normalise_dense_mask(
    mask: Optional[torch.Tensor], layout: AttentionLayout, head_num: int, q: torch.Tensor, v: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return a boolean ``[batch, heads, Sq, Skv]`` mask in the kernel's "``True`` means masked" convention."""
    if mask is None or mask.numel() == 0:
        return None
    if mask.dtype is torch.uint8 or mask.dtype.is_floating_point:
        mask = mask.bool()
    elif mask.dtype is not torch.bool:
        raise TypeError(
            f"attention_mask must be a boolean, uint8 or floating point tensor, got dtype {mask.dtype}. A floating "
            f"point mask is read in the ColossalAI convention, where a non-zero value means masked."
        )

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(
            f"attention_mask must be a 2D [Sq, Skv], 3D [batch, Sq, Skv] or 4D [batch, heads, Sq, Skv] tensor, "
            f"got shape {tuple(mask.shape)}."
        )

    batch = q.size(0)
    sequence_shape = (q.size(layout.sequence_dim), v.size(layout.sequence_dim))
    if mask.size(0) not in (1, batch) or mask.size(1) not in (1, head_num) or tuple(mask.shape[2:]) != sequence_shape:
        raise ValueError(
            f"attention_mask of shape {tuple(mask.shape)} does not match q of shape {tuple(q.shape)} and v of shape "
            f"{tuple(v.shape)} in {layout.name} layout: expected [1 or {batch}, 1 or {head_num}, "
            f"{sequence_shape[0]}, {sequence_shape[1]}]."
        )
    return mask


def _resolve_dense(
    request: _FusionAttentionRequest, layout: AttentionLayout, head_num: int, scale: float
) -> _FusionAttentionArgs:
    if request.cu_seqlens_q is not None or request.cu_seqlens_kv is not None:
        raise ValueError(
            f"The {layout.name} layout is dense and cannot describe variable-length sequences, got "
            f"cu_seqlens_q={request.cu_seqlens_q}, cu_seqlens_kv={request.cu_seqlens_kv}. Pass packed "
            f"[total_tokens, heads, head_dim] tensors with qkv_layout='TND', or keep the padded tensors and provide "
            f"q_indices/kv_indices so that the loaded kernel un-pads them first."
        )

    mask = _normalise_dense_mask(request.attention_mask, layout, head_num, request.q, request.v)
    if mask is not None:
        atten_mask, sparse_mode = mask, _DENSE_MASK_MODE
    elif request.is_causal:
        atten_mask, sparse_mode = _causal_mask(request.q.device), _CAUSAL_WINDOW_MODE
    else:
        atten_mask, sparse_mode = None, _DENSE_MASK_MODE

    return _FusionAttentionArgs(
        head_num=head_num,
        input_layout=layout.name,
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=1.0 - request.dropout_p,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        sparse_mode=sparse_mode,
    )


def _resolve_packed(
    request: _FusionAttentionRequest, layout: AttentionLayout, head_num: int, scale: float
) -> _FusionAttentionArgs:
    if request.attention_mask is not None and request.attention_mask.numel() > 0:
        raise NotImplementedError(
            f"The TND (packed) layout does not accept a dense attention_mask, got shape "
            f"{tuple(request.attention_mask.shape)}. The kernel delimits the sequences with actual_seq_*len, so "
            f"describe them with cu_seqlens_q/cu_seqlens_kv instead, or use a dense layout when a query needs its own "
            f"visibility rule."
        )
    if request.cu_seqlens_q is None:
        raise ValueError(
            "The TND (packed) layout needs cu_seqlens_q=[0, l0, l0 + l1, ...] to delimit the sequences of the "
            "packed tensor; pass it explicitly, even for a single sequence."
        )

    q_lengths = _read_cu_seqlens(request.cu_seqlens_q, "cu_seqlens_q", total_tokens=request.q.size(0))
    if request.cu_seqlens_kv is None:
        if request.k.size(0) != request.q.size(0):
            raise ValueError(
                f"cu_seqlens_kv is required because k/v hold {request.k.size(0)} tokens while q holds "
                f"{request.q.size(0)}; pass the cumulative key/value lengths of the packed batch."
            )
        kv_lengths = q_lengths
    else:
        kv_lengths = _read_cu_seqlens(request.cu_seqlens_kv, "cu_seqlens_kv", total_tokens=request.k.size(0))

    if len(q_lengths) != len(kv_lengths):
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_kv must describe the same number of sequences, got {len(q_lengths)} and "
            f"{len(kv_lengths)}."
        )
    _check_max_seqlen(request.max_seqlen_q, q_lengths, "max_seqlen_q")
    _check_max_seqlen(request.max_seqlen_kv, kv_lengths, "max_seqlen_kv")

    for index, (q_len, kv_len) in enumerate(zip(q_lengths.lengths, kv_lengths.lengths)):
        if q_len > 0 and kv_len == 0:
            raise ValueError(
                f"Sequence {index} has {q_len} query tokens but no key/value tokens; every query needs at least one "
                f"key."
            )
    if request.is_causal:
        for index, (q_len, kv_len) in enumerate(zip(q_lengths.lengths, kv_lengths.lengths)):
            if q_len > kv_len:
                raise ValueError(
                    f"sparse_mode=3 causal attention on the packed TND layout requires every query sequence to be no "
                    f"longer than its key/value sequence, but sequence {index} has {q_len} queries and {kv_len} "
                    f"keys/values. Pad the keys or use a dense layout."
                )
        atten_mask, sparse_mode = _causal_mask(request.q.device), _CAUSAL_WINDOW_MODE
    else:
        atten_mask, sparse_mode = None, _DENSE_MASK_MODE

    return _FusionAttentionArgs(
        head_num=head_num,
        input_layout=layout.name,
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=1.0 - request.dropout_p,
        actual_seq_qlen=q_lengths.actual,
        actual_seq_kvlen=kv_lengths.actual,
        sparse_mode=sparse_mode,
    )


def _resolve(request: _FusionAttentionRequest) -> _FusionAttentionArgs:
    layout = attention_layout(request.qkv_layout)
    head_num, head_dim = _validate_qkv(request, layout)
    scale = request.scale if request.scale is not None else 1.0 / math.sqrt(head_dim)
    if layout.is_packed:
        return _resolve_packed(request, layout, head_num, scale)
    return _resolve_dense(request, layout, head_num, scale)


def resolve_fusion_attention_args(
    q,
    k,
    v,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    attention_mask=None,
    is_causal: bool = False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    qkv_layout: str = "BNSD",
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    head_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Translate the arguments of ``ColoAttention`` into ``torch_npu.npu_fusion_attention`` arguments.

    The translation is device-free apart from building the cached causal mask, so it can be unit tested on CPU. It
    resolves the head count (explicit for ``BSH``, inferred otherwise), normalises a dense mask to the boolean
    "``True`` means masked" form, converts ``cu_seqlens_*`` prefix sums to the cumulative ends the kernel expects,
    and supplies the compressed causal mask when the caller asks for causality without a dense one.

    Args:
        q, k, v: Tensors in ``qkv_layout``.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        scale (Optional[float], optional): Score scale. Defaults to ``head_dim**-0.5``.
        attention_mask (optional): Mask in the ColossalAI convention, where a non-zero value means masked. Boolean
            and uint8 masks are used directly; floating point masks are converted with ``.bool()``. An empty tensor
            is the sentinel that ``prepare_attn_kwargs`` returns when the dense mask would exceed ``MEMORY_BOUND``
            and is read as ``None``. Only dense layouts accept a mask. Defaults to None.
        is_causal (bool, optional): Whether the causal window applies. Without a mask the backend supplies the
            compressed ``[2048, 2048]`` lower-triangular mask and selects ``sparse_mode=3``, because the Ascend
            kernel ignores ``sparse_mode`` when ``atten_mask`` is ``None``. Defaults to False.
        cu_seqlens_q (optional): Cumulative query lengths ``[0, l0, l0 + l1, ...]`` of a packed batch. Required by
            ``TND`` and rejected by the dense layouts. Defaults to None.
        cu_seqlens_kv (optional): Cumulative key/value lengths. When None and ``k`` holds as many tokens as ``q``,
            the query lengths are reused. Defaults to None.
        qkv_layout (str, optional): One of ``QKV_LAYOUTS``. Defaults to "BNSD".
        max_seqlen_q (Optional[int], optional): Largest query length of a packed batch. The compressed mask no
            longer depends on it; when given, it is checked against ``cu_seqlens_q``. Defaults to None.
        max_seqlen_kv (Optional[int], optional): Largest key/value length, checked like ``max_seqlen_q``. Defaults
            to None.
        head_num (Optional[int], optional): Number of heads. Required by ``BSH``, whose last dimension is
            ``heads * head_dim``; for the other layouts it is inferred and, when given, checked against the tensor.
            Defaults to None.

    Returns:
        Dict[str, Any]: Keyword arguments for ``npu_fusion_attention``, plus ``head_num`` and ``input_layout`` which
        that function takes positionally.

    Raises:
        ValueError: If the layout is unknown, the ranks/batch/heads/head_dim disagree, a dense layout is given
            ``cu_seqlens_*``, the lengths are malformed or describe an impossible query/key pairing, or a packed
            causal call has a query sequence longer than its key/value sequence.
        TypeError: If the mask or the cumulative lengths have an unsupported dtype.
        NotImplementedError: If the packed layout is given a non-empty dense mask.

    Note:
        Un-padding by index happens in the loaded kernel, not here: :meth:`FlashAttentionNpuExtension.load` turns a
        padded dense batch plus ``q_indices``/``kv_indices`` into packed tensors and then calls this function with
        ``qkv_layout="TND"``.
    """
    return _resolve(
        _FusionAttentionRequest(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            scale=scale,
            attention_mask=attention_mask,
            is_causal=is_causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            qkv_layout=qkv_layout,
            head_num=head_num,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )
    ).as_dict()


def _flatten_tokens(tensor: torch.Tensor, layout: AttentionLayout, head_num: int) -> torch.Tensor:
    """View a padded dense tensor as ``[total_tokens, heads, head_dim]``, in the order ``get_pad_info`` indexes."""
    if layout.name == "BSH":
        return tensor.reshape(-1, head_num, tensor.size(-1) // head_num)
    if layout.name == "BSND":
        return tensor.reshape(-1, tensor.size(2), tensor.size(3))
    return tensor.transpose(1, 2).reshape(-1, tensor.size(1), tensor.size(3))


def _restore_tokens(
    packed_output: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    layout: AttentionLayout,
) -> torch.Tensor:
    """Scatter ``[total_tokens, heads, head_dim]`` back into the padded dense shape it was un-padded from."""
    num_heads, head_dim = packed_output.size(1), packed_output.size(2)
    padded = packed_output.new_zeros((batch_size * sequence_length, num_heads, head_dim))
    padded.index_copy_(0, indices, packed_output)
    if layout.name == "BSH":
        return padded.reshape(batch_size, sequence_length, num_heads * head_dim)
    padded = padded.view(batch_size, sequence_length, num_heads, head_dim)
    return padded if layout.name == "BSND" else padded.transpose(1, 2)


def _run_fusion_attention(torch_npu, request: _FusionAttentionRequest) -> torch.Tensor:
    arguments = _resolve(request)
    return torch_npu.npu_fusion_attention(
        request.q, request.k, request.v, arguments.head_num, arguments.input_layout, **arguments.kernel_kwargs()
    )[0]


def _attention_over_padded_batch(
    torch_npu,
    request: _FusionAttentionRequest,
    layout: AttentionLayout,
    q_indices: torch.Tensor,
    kv_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """Drop the padding of a padded batch, run the packed path and scatter the output back into the padded shape."""
    head_num, _ = _validate_qkv(request, layout)
    if kv_indices is None:
        kv_indices = q_indices
    batch_size = request.q.size(0)
    sequence_length = request.q.size(layout.sequence_dim)

    packed_request = replace(
        request,
        q=_flatten_tokens(request.q, layout, head_num).index_select(0, q_indices),
        k=_flatten_tokens(request.k, layout, head_num).index_select(0, kv_indices),
        v=_flatten_tokens(request.v, layout, head_num).index_select(0, kv_indices),
        attention_mask=None,
        qkv_layout="TND",
        head_num=None,
    )
    packed_output = _run_fusion_attention(torch_npu, packed_request)
    return _restore_tokens(packed_output, q_indices, batch_size, sequence_length, layout)


class FlashAttentionNpuExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_npu", support_aot=False, support_jit=False)

    def is_available(self) -> bool:
        try:
            import torch_npu

            from colossalai.accelerator import get_accelerator

            return hasattr(torch_npu, "npu_fusion_attention") and get_accelerator().name == "npu"
        except Exception:
            return False

    def assert_compatible(self) -> None:
        pass

    def build_aot(self) -> None:
        raise NotImplementedError(
            "Flash Attention NPU does not require ahead-of-time compilation. Please use it by installing torch_npu."
        )

    def build_jit(self) -> None:
        raise NotImplementedError(
            "Flash Attention NPU does not require just-in-time compilation. Please use it by installing torch_npu."
        )

    def load(self):
        import torch_npu

        def flash_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dropout_p: float = 0.0,
            scale: Optional[float] = None,
            attention_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None,
            q_indices: Optional[torch.Tensor] = None,
            kv_indices: Optional[torch.Tensor] = None,
            qkv_layout: str = "BNSD",
            head_num: Optional[int] = None,
        ):
            """Attention on the NPU in a dense or packed layout.

            A dense padded batch, recognised by ``cu_seqlens_*`` plus ``q_indices``, is un-padded and served by the
            packed ``TND`` path. ``BSH`` fuses the heads into the last dimension, so ``head_num`` is required to
            split it. The kernel derives packed maxima from ``actual_seq_*len``; ``max_seqlen_*`` is only checked
            for consistency.
            """
            layout = attention_layout(qkv_layout)
            request = _FusionAttentionRequest(
                q=q,
                k=k,
                v=v,
                dropout_p=dropout_p,
                scale=scale,
                attention_mask=attention_mask,
                is_causal=is_causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_layout=qkv_layout,
                head_num=head_num,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )
            if layout.is_dense and cu_seqlens_q is not None and q_indices is not None:
                return _attention_over_padded_batch(torch_npu, request, layout, q_indices, kv_indices)
            return _run_fusion_attention(torch_npu, request)

        # A causal call may arrive without a dense mask, either because the caller only asked for causality or
        # because ``prepare_attn_kwargs`` elided an oversized mask. The backend then builds the window itself.
        flash_attention.supports_internal_causal_window = True
        return flash_attention

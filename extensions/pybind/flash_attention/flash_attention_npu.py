"""Flash attention on Ascend NPUs through ``torch_npu.npu_fusion_attention``.

Two input shapes are supported and they are deliberately kept apart. Dense tensors of shape ``[batch, heads, seq,
head_dim]`` (``BNSD``) or ``[batch, seq, heads, head_dim]`` (``BSND``) carry an explicit visibility mask. Packed
tensors of shape ``[total_tokens, heads, head_dim]`` (``TND``) carry :class:`cu_seqlens` instead: the kernel needs the
cumulative sequence lengths to know where one sequence ends and the next one starts, and it builds the causal window
itself, so no ``[seq, seq]`` mask is ever materialised.

A padded batch produced by ``ColoAttention.prepare_attn_kwargs`` arrives in one of the dense layouts together with
``q_indices``/``kv_indices``. Those indices are used to drop the padding and continue on the packed path, which is
what makes the NPU backend save the work that padding would otherwise cause.
"""

import math
from typing import Any, Dict, List, Optional

import torch

from ...base_extension import _Extension

# Layout of the q/k/v tensors as described to ``npu_fusion_attention``. Every entry maps to the index of the head
# dimension inside the tensor shape.
_HEAD_DIM_INDEX = {"BNSD": 1, "BSND": 2, "TND": 1}
_RANK = {"BNSD": 4, "BSND": 4, "TND": 3}

QKV_LAYOUTS = tuple(_HEAD_DIM_INDEX)

# ``sparse_mode`` of ``npu_fusion_attention``: 0 reads the dense mask it is given, 3 is the right-down causal window
# that the kernel builds itself.
_DENSE_MASK_MODE = 0
_CAUSAL_WINDOW_MODE = 3

__all__ = ["FlashAttentionNpuExtension", "resolve_fusion_attention_args", "QKV_LAYOUTS"]


def _validate_cu_seqlens(cu_seqlens, argument_name: str, total_tokens: int) -> torch.Tensor:
    """Check that ``cu_seqlens`` is the prefix sum ``[0, l0, l0 + l1, ...]`` of the packed sequence lengths."""
    lengths = torch.as_tensor(cu_seqlens)
    if lengths.dim() != 1 or lengths.numel() < 2:
        raise ValueError(
            f"{argument_name} must be a 1D prefix sum [0, l0, l0 + l1, ...] describing at least one sequence, got "
            f"{lengths.tolist()} with shape {tuple(lengths.shape)}."
        )
    if lengths.is_floating_point() or lengths.dtype is torch.bool:
        raise TypeError(
            f"{argument_name} must contain integers, got dtype {lengths.dtype}. Cast the lengths with "
            f".to(torch.int32) before calling."
        )
    if int(lengths[0]) != 0:
        raise ValueError(
            f"{argument_name} must start at 0, got {lengths.tolist()}. Pass the cumulative lengths of the packed "
            f"sequence, i.e. [0, l0, l0 + l1, ...]."
        )
    if torch.any(lengths[1:] <= lengths[:-1]):
        raise ValueError(
            f"{argument_name} must be strictly increasing, got {lengths.tolist()}. Every sequence length must be "
            f"positive."
        )
    if int(lengths[-1]) != total_tokens:
        raise ValueError(
            f"{argument_name} must end at the number of tokens of the packed tensor, i.e. {total_tokens}, got "
            f"{int(lengths[-1])} ({lengths.tolist()})."
        )
    return lengths


def _as_actual_seq_lengths(cu_seqlens: torch.Tensor) -> List[int]:
    """Drop the leading zero of a cumulative-length prefix sum, which is the form ``actual_seq_*len`` expects."""
    return [int(length) for length in cu_seqlens[1:].tolist()]


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
) -> Dict[str, Any]:
    """Translate the arguments of the ``ColoAttention`` entry into arguments of ``torch_npu.npu_fusion_attention``.

    The translation is kept free of any device call so that it can be unit tested on CPU. It decides the head count
    and the layout the kernel is told about, picks the dense or the packed path from the layout, normalises the mask
    to the boolean "``True`` means masked" form the NPU kernel expects, and turns the cumulative sequence lengths of
    the packed path into the form the kernel wants.

    Args:
        q: Query tensor in ``qkv_layout``.
        k: Key tensor in ``qkv_layout``.
        v: Value tensor in ``qkv_layout``.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        scale (Optional[float], optional): Multiplier applied to the scores before the softmax. Defaults to
            ``q.size(-1)**-0.5``.
        attention_mask (optional): Mask in the ColossalAI convention, where a **non-zero** value means "masked". A
            floating point mask is converted with ``.bool()``, a boolean mask is passed through, and ``None`` means
            "nothing masked". An **empty** tensor is the sentinel ``ColoAttention.prepare_attn_kwargs`` uses when the
            dense mask would exceed ``MEMORY_BOUND``; it is read as ``None`` as well, and ``is_causal`` then asks the
            kernel to build the window. Only the dense layouts accept a mask. Defaults to None.
        is_causal (bool, optional): Whether a causal mask applies. A causal call without a mask does not build one:
            the kernel receives ``sparse_mode=3`` and generates the right-down causal window itself. Defaults to
            False.
        cu_seqlens_q (optional): Cumulative query lengths of a packed batch, i.e. ``[0, l0, l0 + l1, ...]``. The
            leading zero is dropped before the lengths reach the kernel, which expects the cumulative ends. Required
            by ``TND`` and rejected by the dense layouts. Defaults to None.
        cu_seqlens_kv (optional): Cumulative key/value lengths of a packed batch. When it is None and ``k`` holds as
            many tokens as ``q``, the query lengths are reused. Defaults to None.
        qkv_layout (str, optional): Memory layout of ``q``, ``k`` and ``v``, one of ``QKV_LAYOUTS``. ``BNSD`` is
            ``[batch, heads, sequence, head_dim]``, ``BSND`` is ``[batch, sequence, heads, head_dim]`` and ``TND`` is
            the packed ``[total_tokens, heads, head_dim]``. The tensors must already be contiguous in that layout,
            because re-ordering them here would reintroduce the copy this option exists to avoid. Defaults to "BNSD".

    Returns:
        Dict[str, Any]: Keyword arguments for ``npu_fusion_attention``, plus ``head_num`` and ``input_layout`` which
        that function takes positionally, and ``sparse_mode`` which selects between the given mask and the window
        the kernel builds itself.

    Raises:
        ValueError: If ``qkv_layout`` is unknown, if the tensor rank does not match the layout, if the shapes disagree
            on the head count, if a dense layout is handed ``cu_seqlens_*``, or if ``cu_seqlens_*`` is not a prefix
            sum ending at the number of packed tokens.
        TypeError: If ``attention_mask`` is a tensor of a dtype that cannot be interpreted as a mask, or if
            ``cu_seqlens_*`` is not integral.
        NotImplementedError: If the packed layout is handed a dense mask, which cannot be delimited by
            ``actual_seq_*len``.

    Note:
        Un-padding by index happens in the loaded kernel, not here: :meth:`FlashAttentionNpuExtension.load` turns a
        padded dense batch plus ``q_indices``/``kv_indices`` into packed tensors and then calls this function with
        ``qkv_layout="TND"``.
    """
    if qkv_layout not in _RANK:
        raise ValueError(f"Unsupported qkv_layout {qkv_layout!r}; expected one of {list(QKV_LAYOUTS)}.")

    expected_rank = _RANK[qkv_layout]
    if q.dim() != expected_rank or k.dim() != expected_rank or v.dim() != expected_rank:
        raise ValueError(
            f"q, k and v must be {expected_rank}D tensors in {qkv_layout} layout, got shapes "
            f"{tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
        )

    head_dim_index = _HEAD_DIM_INDEX[qkv_layout]
    if q.shape[head_dim_index] != k.shape[head_dim_index] or k.shape[head_dim_index] != v.shape[head_dim_index]:
        raise ValueError(
            f"q, k and v must have the same number of heads in {qkv_layout} layout (dimension {head_dim_index}), got "
            f"shapes {tuple(q.shape)}, {tuple(k.shape)} and {tuple(v.shape)}."
        )
    head_num = q.size(head_dim_index)

    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    if attention_mask is not None and attention_mask.numel() == 0:
        attention_mask = None

    if qkv_layout == "TND":
        return _resolve_packed_arguments(
            q, k, v, head_num, scale, dropout_p, attention_mask, is_causal, cu_seqlens_q, cu_seqlens_kv
        )
    return _resolve_dense_arguments(
        q, k, v, head_num, scale, dropout_p, attention_mask, is_causal, cu_seqlens_q, cu_seqlens_kv, qkv_layout
    )


def _resolve_dense_arguments(
    q,
    k,
    v,
    head_num: int,
    scale: float,
    dropout_p: float,
    attention_mask,
    is_causal: bool,
    cu_seqlens_q,
    cu_seqlens_kv,
    qkv_layout: str,
) -> Dict[str, Any]:
    """Arguments of the dense ``BNSD``/``BSND`` path, where every query carries its own visibility row."""
    if cu_seqlens_q is not None or cu_seqlens_kv is not None:
        raise ValueError(
            f"The {qkv_layout} layout is dense and cannot describe variable-length sequences, got "
            f"cu_seqlens_q={cu_seqlens_q}, cu_seqlens_kv={cu_seqlens_kv}. Pass packed [total_tokens, heads, head_dim] "
            f"tensors with qkv_layout='TND', or keep the padded tensors and provide q_indices/kv_indices so that the "
            f"loaded kernel un-pads them first."
        )

    if attention_mask is None:
        atten_mask = None
        sparse_mode = _CAUSAL_WINDOW_MODE if is_causal else _DENSE_MASK_MODE
    elif attention_mask.dtype is torch.bool or attention_mask.is_floating_point():
        if attention_mask.dim() == 2:
            mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4:
            mask = attention_mask
        else:
            raise ValueError(
                f"attention_mask must be a 2D [Sq, Skv], 3D [batch, Sq, Skv] or 4D [batch, 1, Sq, Skv] tensor, got "
                f"shape {tuple(attention_mask.shape)}."
            )
        atten_mask = mask if mask.dtype is torch.bool else mask.bool()
        sparse_mode = _DENSE_MASK_MODE
    else:
        raise TypeError(
            f"attention_mask must be a boolean or floating point tensor, got dtype {attention_mask.dtype}. A "
            f"floating point mask is read in the ColossalAI convention, where a non-zero value means masked."
        )

    return {
        "head_num": head_num,
        "input_layout": qkv_layout,
        "atten_mask": atten_mask,
        "scale": scale,
        "keep_prob": 1.0 - dropout_p,
        "actual_seq_qlen": None,
        "actual_seq_kvlen": None,
        "sparse_mode": sparse_mode,
    }


def _resolve_packed_arguments(
    q,
    k,
    v,
    head_num: int,
    scale: float,
    dropout_p: float,
    attention_mask,
    is_causal: bool,
    cu_seqlens_q,
    cu_seqlens_kv,
) -> Dict[str, Any]:
    """Arguments of the packed ``TND`` path, where the sequences are delimited by ``actual_seq_*len``."""
    if attention_mask is not None:
        raise NotImplementedError(
            f"The TND (packed) layout does not accept a dense attention_mask, got shape "
            f"{tuple(attention_mask.shape)}. The kernel delimits the sequences with actual_seq_*len, so describe them "
            f"with cu_seqlens_q/cu_seqlens_kv instead, or use qkv_layout='BNSD'/'BSND' when a query needs its own "
            f"visibility rule."
        )
    if k.size(0) != v.size(0):
        raise ValueError(f"k and v must hold the same number of packed tokens, got {k.size(0)} and {v.size(0)}.")
    if cu_seqlens_q is None:
        raise ValueError(
            f"The TND (packed) layout needs cu_seqlens_q=[0, l0, l0 + l1, ...] to delimit the sequences of the "
            f"packed tensor; pass it explicitly, even for a single sequence."
        )

    cu_seqlens_q = _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", total_tokens=q.size(0))
    if cu_seqlens_kv is None:
        if k.size(0) != q.size(0):
            raise ValueError(
                f"cu_seqlens_kv is required because k/v hold {k.size(0)} tokens while q holds {q.size(0)}; pass the "
                f"cumulative key/value lengths of the packed batch."
            )
        cu_seqlens_kv = cu_seqlens_q
    else:
        cu_seqlens_kv = _validate_cu_seqlens(cu_seqlens_kv, "cu_seqlens_kv", total_tokens=k.size(0))
    if len(cu_seqlens_q) != len(cu_seqlens_kv):
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_kv must describe the same number of sequences, got "
            f"{len(cu_seqlens_q) - 1} and {len(cu_seqlens_kv) - 1}."
        )

    return {
        "head_num": head_num,
        "input_layout": "TND",
        "atten_mask": None,
        "scale": scale,
        "keep_prob": 1.0 - dropout_p,
        "actual_seq_qlen": _as_actual_seq_lengths(cu_seqlens_q),
        "actual_seq_kvlen": _as_actual_seq_lengths(cu_seqlens_kv),
        "sparse_mode": _CAUSAL_WINDOW_MODE if is_causal else _DENSE_MASK_MODE,
    }


def _flatten_tokens(tensor: torch.Tensor, qkv_layout: str) -> torch.Tensor:
    """View a padded dense tensor as ``[total_tokens, heads, head_dim]``, in the order ``get_pad_info`` indexes."""
    if qkv_layout == "BSND":
        return tensor.reshape(-1, tensor.size(2), tensor.size(3))
    return tensor.transpose(1, 2).reshape(-1, tensor.size(1), tensor.size(3))


def _restore_tokens(
    packed_output: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    qkv_layout: str,
) -> torch.Tensor:
    """Scatter ``[total_tokens, heads, head_dim]`` back into the padded dense shape it was un-padded from."""
    num_heads, head_dim = packed_output.size(1), packed_output.size(2)
    padded = packed_output.new_zeros((batch_size * sequence_length, num_heads, head_dim))
    padded.index_copy_(0, indices, packed_output)
    padded = padded.view(batch_size, sequence_length, num_heads, head_dim)
    return padded if qkv_layout == "BSND" else padded.transpose(1, 2)


def _run_fusion_attention(
    torch_npu,
    q,
    k,
    v,
    dropout_p: float,
    scale: Optional[float],
    attention_mask,
    is_causal: bool,
    cu_seqlens_q,
    cu_seqlens_kv,
    qkv_layout: str,
) -> torch.Tensor:
    """Resolve the arguments and call the kernel, dropping the ones this path leaves unset."""
    arguments = resolve_fusion_attention_args(
        q,
        k,
        v,
        dropout_p=dropout_p,
        scale=scale,
        attention_mask=attention_mask,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        qkv_layout=qkv_layout,
    )
    head_num = arguments.pop("head_num")
    input_layout = arguments.pop("input_layout")
    arguments = {name: value for name, value in arguments.items() if value is not None}
    return torch_npu.npu_fusion_attention(q, k, v, head_num, input_layout, **arguments)[0]


def _attention_over_padded_batch(
    torch_npu,
    q,
    k,
    v,
    dropout_p: float,
    scale: Optional[float],
    is_causal: bool,
    cu_seqlens_q,
    cu_seqlens_kv,
    q_indices,
    kv_indices,
    qkv_layout: str,
) -> torch.Tensor:
    """Drop the padding of a padded batch, run the packed path and scatter the output back into the padded shape."""
    if kv_indices is None:
        kv_indices = q_indices
    batch_size = q.size(0)
    sequence_length = q.size(2) if qkv_layout == "BNSD" else q.size(1)

    packed_q = _flatten_tokens(q, qkv_layout).index_select(0, q_indices)
    packed_k = _flatten_tokens(k, qkv_layout).index_select(0, kv_indices)
    packed_v = _flatten_tokens(v, qkv_layout).index_select(0, kv_indices)

    packed_output = _run_fusion_attention(
        torch_npu,
        packed_q,
        packed_k,
        packed_v,
        dropout_p=dropout_p,
        scale=scale,
        attention_mask=None,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        qkv_layout="TND",
    )
    return _restore_tokens(packed_output, q_indices, batch_size, sequence_length, qkv_layout)


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
        ):
            """Attention on the NPU, in either the dense or the packed layout.

            A padded dense batch, i.e. a 4D call that carries ``cu_seqlens_*`` and ``q_indices``, is un-padded first
            and then served by the packed path. ``max_seqlen_q``/``max_seqlen_kv`` are not used because the kernel
            derives the maxima from ``actual_seq_*len``.
            """
            if q.dim() == 4 and qkv_layout in ("BNSD", "BSND") and cu_seqlens_q is not None and q_indices is not None:
                return _attention_over_padded_batch(
                    torch_npu,
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    scale=scale,
                    is_causal=is_causal,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    q_indices=q_indices,
                    kv_indices=kv_indices,
                    qkv_layout=qkv_layout,
                )
            return _run_fusion_attention(
                torch_npu,
                q,
                k,
                v,
                dropout_p=dropout_p,
                scale=scale,
                attention_mask=attention_mask,
                is_causal=is_causal,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                qkv_layout=qkv_layout,
            )

        # A causal call without a mask is served by ``sparse_mode=3``, so an omitted
        # ``[batch, 1, seq, seq]`` mask is not a missing capability here.
        flash_attention.supports_internal_causal_window = True
        return flash_attention

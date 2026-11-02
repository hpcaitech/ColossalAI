"""Layout descriptors shared by the attention entry and its backends."""

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["AttentionLayout", "DENSE_QKV_LAYOUTS", "QKV_LAYOUTS", "attention_layout"]


@dataclass(frozen=True)
class AttentionLayout:
    """How a q/k/v tensor is laid out.

    Attributes:
        name (str): Name of the layout, as passed to the attention backends.
        rank (int): Number of dimensions of one tensor.
        sequence_dim (Optional[int]): Dimension that holds the sequence, or ``None`` for a packed tensor whose
            sequences are delimited by cumulative lengths.
        head_dim_index (Optional[int]): Dimension that holds the number of heads, or ``None`` when the heads are
            fused into the last dimension (``BSH``), where only ``heads * head_dim`` is visible.
    """

    name: str
    rank: int
    sequence_dim: Optional[int]
    head_dim_index: Optional[int]

    @property
    def is_dense(self) -> bool:
        return self.sequence_dim is not None

    @property
    def is_packed(self) -> bool:
        return not self.is_dense

    @property
    def has_fused_heads(self) -> bool:
        return self.head_dim_index is None


_LAYOUTS = {
    "BNSD": AttentionLayout("BNSD", rank=4, sequence_dim=-2, head_dim_index=1),
    "BSND": AttentionLayout("BSND", rank=4, sequence_dim=-3, head_dim_index=2),
    "BSH": AttentionLayout("BSH", rank=3, sequence_dim=1, head_dim_index=None),
    "TND": AttentionLayout("TND", rank=3, sequence_dim=None, head_dim_index=1),
}

QKV_LAYOUTS: Tuple[str, ...] = tuple(_LAYOUTS)
DENSE_QKV_LAYOUTS: Tuple[str, ...] = tuple(name for name, layout in _LAYOUTS.items() if layout.is_dense)


def attention_layout(name: str) -> AttentionLayout:
    """Return the descriptor of ``name``.

    Raises:
        ValueError: If ``name`` is not a known layout.
    """
    try:
        return _LAYOUTS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported qkv_layout {name!r}; expected one of {list(QKV_LAYOUTS)}. The sequence-first 'SBH' "
            f"layout is intentionally unsupported because no ColossalAI path produces it."
        ) from None

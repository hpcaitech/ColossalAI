"""Variable-length attention on NVIDIA GPUs through PyTorch's native ``varlen_attn``.

PyTorch 2.13 ships ``torch.nn.attention.varlen.varlen_attn``, which is a flash-attention backed kernel over
``[total_tokens, heads, head_dim]`` tensors with cumulative sequence lengths. It serves the two calls a Cosmos-style
two-way attention needs without materialising an ``[S, S]`` mask: a causal call delimited by the AR boundaries and a
full call whose K/V boundaries describe each sample's ``[AR_i; DM_i]`` block.

The extension is only a thin adapter: it keeps the loader and the two-way layer free of a direct torch import and
reports whether the pinned Torch build exposes the kernel.
"""

import importlib

from ...base_extension import _Extension


class FlashAttentionVarlenCudaExtension(_Extension):
    """Expose ``torch.nn.attention.varlen.varlen_attn`` as a ColossalAI kernel."""

    def __init__(self):
        super().__init__(name="flash_attention_varlen_cuda", support_aot=False, support_jit=False, priority=20)

    def is_available(self) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            varlen_module = importlib.import_module("torch.nn.attention.varlen")
        except Exception:
            return False
        return hasattr(varlen_module, "varlen_attn")

    def assert_compatible(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("native varlen attention requires a visible CUDA device")
            varlen_module = importlib.import_module("torch.nn.attention.varlen")
        except ImportError as error:
            raise RuntimeError(
                "This PyTorch build does not provide torch.nn.attention.varlen.varlen_attn. Install a CUDA build of "
                "PyTorch 2.10 or newer, or use the naive/SDPA two-way backend for correctness checks."
            ) from error
        if not hasattr(varlen_module, "varlen_attn"):
            raise RuntimeError(
                "This PyTorch build does not provide torch.nn.attention.varlen.varlen_attn. Install a CUDA build of "
                "PyTorch 2.10 or newer, or use the naive/SDPA two-way backend for correctness checks."
            )

    def build_aot(self) -> None:
        raise NotImplementedError("Native varlen attention does not require ahead-of-time compilation.")

    def build_jit(self) -> None:
        raise NotImplementedError("Native varlen attention does not require just-in-time compilation.")

    def load(self):
        from typing import Optional as _Optional

        import torch
        from torch.nn.attention.varlen import varlen_attn

        def flash_attention_varlen(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_kv: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_kv: int,
            is_causal: bool = False,
            scale: _Optional[float] = None,
            enable_gqa: bool = False,
        ) -> torch.Tensor:
            """Packed attention over ``[total_tokens, heads, head_dim]`` tensors.

            Args:
                q (torch.Tensor): Query tensor of shape ``[total_q_tokens, heads, head_dim]``.
                k (torch.Tensor): Key tensor of shape ``[total_kv_tokens, kv_heads, head_dim]``.
                v (torch.Tensor): Value tensor of shape ``[total_kv_tokens, kv_heads, head_dim]``.
                cu_seqlens_q (torch.Tensor): Prefix sums of the query lengths, shape ``[samples + 1]``.
                cu_seqlens_kv (torch.Tensor): Prefix sums of the key/value lengths, shape ``[samples + 1]``.
                max_seqlen_q (int): Longest query length of the batch.
                max_seqlen_kv (int): Longest key/value length of the batch.
                is_causal (bool, optional): Apply the lower-triangular window. Defaults to False.
                scale (_Optional[float], optional): Softmax scale. Defaults to None, i.e. ``head_dim**-0.5``.
                enable_gqa (bool, optional): Let ``heads`` be a multiple of ``kv_heads``. Defaults to False.

            Returns:
                torch.Tensor: Output of shape ``[total_q_tokens, heads, head_dim]``.
            """
            window_size = (-1, 0) if is_causal else (-1, -1)
            return varlen_attn(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                window_size=window_size,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        # The caller selects this kernel explicitly for packed tensors; it never receives a dense mask.
        flash_attention_varlen.supports_packed_varlen = True
        return flash_attention_varlen

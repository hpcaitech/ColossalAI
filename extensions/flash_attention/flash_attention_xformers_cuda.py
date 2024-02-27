from ..base_extension import _Extension


class FlashAttentionXformersCudaExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_xformers_cuda", support_aot=False, support_jit=False)

    def is_hardware_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def assert_hardware_compatible(self) -> bool:
        pass

    def build_aot(self) -> None:
        raise NotImplementedError(
            "We rely on the third-party xformers library for flash attention (https://github.com/facebookresearch/xformers). Please install xformers according to the GitHub Readme."
        )

    def build_jit(self) -> None:
        raise NotImplementedError(
            "We rely on the third-party xformers library for flash attention (https://github.com/facebookresearch/xformers). Please install xformers according to the GitHub Readme."
        )

    def load(self):
        try:
            from xformers.ops.fmha import MemoryEfficientAttentionCutlassOp, memory_efficient_attention
            from xformers.ops.fmha.attn_bias import (
                BlockDiagonalCausalMask,
                BlockDiagonalMask,
                LowerTriangularMask,
                LowerTriangularMaskWithTensorBias,
            )
        except ImportError:
            raise ModuleNotFoundError(
                (
                    "We rely on the third-party xformers library for flash attention (https://github.com/facebookresearch/xformers). Please install xformers according to the GitHub Readme."
                )
            )
        from typing import Optional

        import torch

        allow_alibi = True
        for op in MemoryEfficientAttentionCutlassOp:
            allow_alibi = allow_alibi & (LowerTriangularMaskWithTensorBias in op.SUPPORTED_ATTN_BIAS_TYPES)

        def mem_eff_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_len_info_q: "SeqLenInfo",
            seq_len_info_kv: "SeqLenInfo",
            origin_attn_mask: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
            scale: float = None,
            causal: bool = False,
            padded: bool = False,
        ):
            attn_bias = None
            if padded:  # bert style
                if not causal:
                    attn_bias = BlockDiagonalMask.from_seqlens(seq_len_info_q.seqlens, seq_len_info_kv.seqlens)
                else:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_len_info_q.seqlens, seq_len_info_kv.seqlens)
            elif causal:  # gpt style
                attn_bias = LowerTriangularMask()

            if bias is not None:  # alibi / relative position embedding
                assert allow_alibi, "flash attention with bias is not supported in this system."
                assert causal, "attention with bias is only supported for causal attention so far."
                attn_bias = attn_bias.add_bias(bias)

            if padded:
                q = q.unsqueeze(0)
                k = k.unsqueeze(0)
                v = v.unsqueeze(0)

            out = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=dropout_p, scale=scale)

            # shape: (b*s, n, d)
            if padded:
                out = out.squeeze(0)

            return out

        return mem_eff_attention

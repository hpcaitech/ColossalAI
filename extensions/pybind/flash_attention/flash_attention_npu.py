import math

from ...base_extension import _Extension


class FlashAttentionNpuExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_npu", support_aot=False, support_jit=False)

    def is_available(self) -> bool:
        try:
            import torch_npu

            return hasattr(torch_npu, "npu_fusion_attention")
        except:
            return False

    def assert_compatible(self) -> bool:
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
        from typing import Optional

        import torch
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
        ):
            if scale is None:
                scale = 1.0 / math.sqrt(q.size(-1))
            num_heads = q.size(1)
            return torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                num_heads,
                "BNSD",
                atten_mask=attention_mask.bool(),
                scale=scale,
                keep_prob=1 - dropout_p,
            )[0]

        return flash_attention

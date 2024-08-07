from ...base_extension import _Extension


class FlashAttentionDaoCudaExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_dao_cuda", support_aot=False, support_jit=False, priority=10)

    def is_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            from flash_attn import flash_attn_func, flash_attn_varlen_kvpacked_func  # noqa
            from flash_attn.bert_padding import index_first_axis, pad_input  # noqa

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def assert_compatible(self) -> bool:
        pass

    def build_aot(self) -> None:
        raise NotImplementedError(
            "We rely on the third-party flash-attn library for flash attention (https://github.com/Dao-AILab/flash-attention). Please install flash-attn via 'pip install flash-attn --no-build-isolation'."
        )

    def build_jit(self) -> None:
        raise NotImplementedError(
            "We rely on the third-party flash-attn library for flash attention (https://github.com/Dao-AILab/flash-attention). Please install flash-attn via 'pip install flash-attn --no-build-isolation'"
        )

    def load(self):
        from typing import Optional

        import torch
        from einops import rearrange
        from flash_attn import flash_attn_func, flash_attn_varlen_kvpacked_func
        from flash_attn.bert_padding import index_first_axis, pad_input

        def _unpad_input(hidden_states: torch.Tensor, indices: torch.Tensor):
            return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)

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
            # [B, H, S, D] -> [B, S, H, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            b, s_q = q.shape[:2]
            if cu_seqlens_q is not None:
                # padded / padded causal
                # unpad input: [B, S, H, D] -> [T, H, D]
                q = _unpad_input(q, q_indices)
                kv = _unpad_input(torch.stack(tensors=(k, v), dim=2), kv_indices)
                attn_output = flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=is_causal,
                )
                # pad output: [T, H, D] -> [B, S, H, D]
                attn_output = pad_input(attn_output, q_indices, b, s_q)
            else:
                # causal / no attn mask
                attn_output = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=is_causal,
                )
            # [B, S, H, D] -> [B, H, S, D]
            return attn_output.transpose(1, 2)

        return flash_attention

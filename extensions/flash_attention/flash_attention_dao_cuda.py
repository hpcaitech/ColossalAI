from ..base_extension import _Extension


class FlashAttentionDaoCudaExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_dao_cuda", support_aot=False, support_jit=False, priority=10)

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
            "We rely on the third-party flash-attn library for flash attention (https://github.com/Dao-AILab/flash-attention). Please install flash-attn via 'pip install flash-attn --no-build-isolation'."
        )

    def build_jit(self) -> None:
        raise NotImplementedError(
            "We rely on the third-party flash-attn library for flash attention (https://github.com/Dao-AILab/flash-attention). Please install flash-attn via 'pip install flash-attn --no-build-isolation'"
        )

    def load(self):
        try:
            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        except ImportError:
            raise ModuleNotFoundError(
                (
                    "We rely on the third-party flash-attn library for flash attention. Please install flash-attn via 'pip install flash-attn --no-build-isolation'"
                )
            )

        from typing import Optional

        import torch

        def flash_attention(
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
            """
            Arguments:
                q: (batch, q_seqlen, nheads, headdim)
                k: (batch, kv_seqlen, nheads, headdim)
                v: (batch, kv_seqlen, nheads, headdim)
                batch_size: int.
                seq_len: int.
                dropout_p: float. Dropout probability.
                sm_scale: float. The scaling of QK^T before applying softmax.
                    Default to 1 / sqrt(headdim).
                causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            Return:
                attn_out: (batch, q_seqlen, nheads, headdim).
            """
            # check if the input is in allowed dtypes
            if padded:
                if seq_len_info_kv == None:
                    seq_len_info_kv = seq_len_info_q

                attn_out = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    seq_len_info_q.cu_seqlens,
                    seq_len_info_kv.cu_seqlens,
                    seq_len_info_q.max_seqlen,
                    seq_len_info_kv.max_seqlen,
                    dropout_p,
                    scale,
                    causal,
                )
            else:
                attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=causal)
            return attn_out

        return flash_attention

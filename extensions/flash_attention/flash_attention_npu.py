from ..base_extension import _Extension


class FlashAttentionNpuExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_npu", support_aot=False, support_jit=False)

    def is_hardware_available(self) -> bool:
        try:
            import torch_npu  # noqa

            return True
        except:
            return False

    def assert_hardware_compatible(self) -> bool:
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
        import torch
        from einops import rearrange

        def npu_sdpa_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_len_info_q=None,
            seq_len_info_kv=None,
            origin_attn_mask: torch.Tensor = None,
            dropout_p: float = 0.0,
            scale: float = 1.0,
            causal=None,
            padded=None,
        ):
            """
            The scaled dot product attention.

            Arguments:
                q: (batch, q_seqlen, nheads, headdim)
                k: (batch, kv_seqlen, nheads, headdim)
                v: (batch, kv_seqlen, nheads, headdim)
                batch_size: int.
                seq_len: int.
                dropout_p: float. Dropout probability.
                scale: float. The scaling of QK^T before applying softmax.
                    Default to 1.
            Return:
                attn_out: (batch, q_seqlen, nheads, headdim).
            """
            q, k, v = [rearrange(x, "b s h d -> b h s d").contiguous() for x in (q, k, v)]
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=origin_attn_mask,
                dropout_p=dropout_p,
                is_causal=origin_attn_mask is None,
                scale=scale,
            )
            output = rearrange(output, "b h s d -> b s (h d)")
            return output

        return npu_sdpa_attention

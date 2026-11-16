from importlib.util import find_spec

from ...base_extension import _Extension


def _output_tensor(result):
    import torch

    # Older Hopper APIs return (out, softmax_lse). Keep out's autograd graph.
    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError("FA3 must return a Tensor or (Tensor, softmax_lse)")
    return result


class FlashAttentionDaoV3CudaExtension(_Extension):
    """FA3 training API at Dao-AILab/flash-attention commit 54b044c8cf897dfcc2577a665559d1bfda0bfc80.

    Install the ``hopper`` subdirectory of that revision as ``flash_attn_3``.
    This API has no dropout argument. It is deliberately not registered in the
    legacy automatic dispatcher, so existing callers keep their backend choice.
    """

    def __init__(self):
        super().__init__(name="flash_attention_dao_v3_cuda", support_aot=False, support_jit=False)

    def is_available(self) -> bool:
        import torch

        return torch.version.hip is None and torch.cuda.is_available() and find_spec("flash_attn_3") is not None

    def assert_compatible(self) -> None:
        pass  # Request-specific device/dtype checks belong to ColoAttention.

    def build_aot(self) -> None:
        raise NotImplementedError("Install the FA3 hopper package from Dao-AILab/flash-attention separately.")

    def build_jit(self) -> None:
        self.build_aot()

    def load(self):
        from flash_attn_3 import flash_attn_interface

        def flash_attention(
            q, k, v, dropout_p=0.0, scale=None, attention_mask=None, is_causal=False, causal_alignment=None
        ):
            if dropout_p != 0 or attention_mask is not None:
                raise ValueError("FA3 does not support dropout or an arbitrary attention mask")
            # A transpose is a view. Copy only when the kernel's last-dimension
            # stride requirement is not met; keep the autograd connection.
            inputs = [x.transpose(1, 2) for x in (q, k, v)]
            inputs = [x if x.stride(-1) == 1 else x.contiguous() for x in inputs]
            result = flash_attn_interface.flash_attn_func(*inputs, softmax_scale=scale, causal=is_causal)
            return _output_tensor(result).transpose(1, 2)

        return flash_attention

    def load_varlen(self):
        from flash_attn_3 import flash_attn_interface

        def flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p=0.0,
            scale=None,
            is_causal=False,
            causal_alignment=None,
        ):
            if dropout_p != 0:
                raise ValueError("FA3 does not support dropout")
            inputs = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]
            result = flash_attn_interface.flash_attn_varlen_func(
                *inputs,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                softmax_scale=scale,
                causal=is_causal,
            )
            return _output_tensor(result)

        return flash_attention_varlen

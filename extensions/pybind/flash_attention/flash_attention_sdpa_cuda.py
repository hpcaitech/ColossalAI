from ...base_extension import _Extension


class FlashAttentionSdpaCudaExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_sdpa_cuda", support_aot=False, support_jit=False)

    def is_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def assert_compatible(self) -> bool:
        pass

    def build_aot(self) -> None:
        raise NotImplementedError("Flash attention SDPA does not require ahead-of-time compilation.")

    def build_jit(self) -> None:
        raise NotImplementedError("Flash attention SDPA does not require just-in-time compilation.")

    def load(self, strict=False):
        from contextlib import nullcontext
        from typing import Optional

        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

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
            kwargs = {}
            if strict:
                # Legacy calls already encode causal attention in attention_mask.
                # The new explicit API can use native causal attention and GQA.
                kwargs["is_causal"] = is_causal
                if q.shape[1] != k.shape[1]:
                    kwargs["enable_gqa"] = True
            context = nullcontext()
            if (
                strict
                and q.is_cuda
                and q.dtype in (torch.float16, torch.bfloat16)
                and attention_mask is not None
                and attention_mask.dtype == torch.float32
            ):
                # cuDNN SDPA in Torch 2.13 can produce NaNs or incorrect results
                # for fp32 additive masks with half inputs. Preserve the mask's
                # precision and the caller's other enabled SDPA kernels.
                candidates = (
                    (SDPBackend.FLASH_ATTENTION, torch.backends.cuda.flash_sdp_enabled()),
                    (SDPBackend.EFFICIENT_ATTENTION, torch.backends.cuda.mem_efficient_sdp_enabled()),
                    (SDPBackend.MATH, torch.backends.cuda.math_sdp_enabled()),
                )
                context = sdpa_kernel([kernel for kernel, enabled in candidates if enabled])
            with context:
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attention_mask,
                    dropout_p=dropout_p,
                    scale=scale,
                    **kwargs,
                )

        return flash_attention

    def load_varlen(self):
        import torch

        dense_attention = self.load(strict=True)

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
            # Correctness fallback, not a fused varlen implementation. Slice each
            # sample independently so no token can attend across sample boundaries.
            q_bounds = cu_seqlens_q.tolist()
            kv_bounds = cu_seqlens_kv.tolist()
            outputs = []
            for q_start, q_end, kv_start, kv_end in zip(q_bounds[:-1], q_bounds[1:], kv_bounds[:-1], kv_bounds[1:]):
                qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
                ki = k[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
                vi = v[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
                mask = None
                native_causal = is_causal
                if is_causal and causal_alignment == "lower_right" and q_end - q_start != kv_end - kv_start:
                    sq, sk = q_end - q_start, kv_end - kv_start
                    mask = torch.arange(sk, device=q.device)[None, :] <= (
                        torch.arange(sq, device=q.device)[:, None] + sk - sq
                    )
                    native_causal = False
                out = dense_attention(
                    qi,
                    ki,
                    vi,
                    dropout_p=dropout_p,
                    scale=scale,
                    attention_mask=mask,
                    is_causal=native_causal,
                )
                outputs.append(out.squeeze(0).transpose(0, 1))
            return torch.cat(outputs, dim=0)

        return flash_attention_varlen

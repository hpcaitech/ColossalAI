import torch
import warnings


HAS_NPU_FUSED_ATTN = False
try:
    from torch_npu import npu_fusion_attention

    HAS_NPU_FUSED_ATTN = True
except ImportError:
    warnings.warn("please install torch_npu with npu_fusion_attention")


if HAS_NPU_FUSED_ATTN:

    def npu_fused_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        scale: float = 1.0,
        dropout_p: float = 0.0,
    ):
        """
        Implement the scaled dot product attention with softmax.

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
        batch, q_len, num_heads = q.shape[:3]
        kv_len = k.shape[1]
        matmul_result = torch.empty(
            (batch, num_heads, q_len, kv_len), dtype=q.dtype, device=q.device
        )
        output = npu_fusion_attention(
            query=q,
            key=k,
            value=v,
            head_num=num_heads,
            input_layout="BSH",
            pse=matmul_result,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=scale,
            pre_tockens=kv_len,
            next_tockens=0,
            keep_prob=1 - dropout_p,
        )[0]
        return output


    if __name__ == "__main__":
        b, s, h, d = 4, 32, 16, 64
        q, k, v = [torch.rand(b, s, h * d).npu() for _ in range(3)]
        context_layer = npu_fused_attention(q, k, v, None)
        print(context_layer)

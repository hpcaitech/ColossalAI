from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from colossalai.inference.config import ModelShardInferenceConfig
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import context_attention_unpadded, flash_decoding_attention


@dataclass
class AttentionMetaData:
    query_states: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    block_size: int
    kv_seq_len: int = None
    sequence_lengths: torch.Tensor = None
    cu_seqlens: torch.Tensor = None
    sm_scale: int = None
    alibi_slopes: torch.Tensor = None
    output_tensor: torch.Tensor = None
    use_spec_dec: bool = False
    use_alibi_attn: bool = False


class AttentionBackend(ABC):
    @abstractmethod
    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, attn_metadatas: AttentionMetaData, **kwargs):
        raise NotImplementedError


class CudaAttentionBackend(AttentionBackend):
    """
    Attention backend when use_cuda_kernel is True but flash-attn not found. If flash-attn is not found,
    it uses Triton op `context_attention_unpadded` for prefilling and our cuda op `flash_decoding_attention` for decoding.
    """

    def __init__(self, use_flash_attn: bool = False):
        super().__init__()
        self.inference_ops = InferenceOpsLoader().load()
        self.use_flash_attn = use_flash_attn

    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        if self.use_flash_attn:
            token_nums = kwargs.get("token_nums", -1)

            from flash_attn import flash_attn_varlen_func

            attn_output = flash_attn_varlen_func(
                attn_metadata.query_states,
                attn_metadata.key_states,
                attn_metadata.value_states,
                cu_seqlens_q=attn_metadata.cu_seqlens,
                cu_seqlens_k=attn_metadata.cu_seqlens,
                max_seqlen_q=attn_metadata.kv_seq_len,
                max_seqlen_k=attn_metadata.kv_seq_len,
                dropout_p=0.0,
                softmax_scale=attn_metadata.sm_scale,
                causal=True,
                alibi_slopes=attn_metadata.alibi_slopes,
            )
            attn_output = attn_output.view(token_nums, -1)
        else:
            attn_output = context_attention_unpadded(
                q=attn_metadata.query_states,
                k=attn_metadata.key_states,
                v=attn_metadata.value_states,
                k_cache=attn_metadata.k_cache,
                v_cache=attn_metadata.v_cache,
                context_lengths=attn_metadata.sequence_lengths,
                block_tables=attn_metadata.block_tables,
                block_size=attn_metadata.block_size,
                output=attn_metadata.output_tensor,
                alibi_slopes=attn_metadata.alibi_slopes,
                max_seq_len=attn_metadata.kv_seq_len,
                sm_scale=attn_metadata.sm_scale,
                use_new_kcache_layout=True,  # use new k-cache layout
            )
        return attn_output

    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        fd_inter_tensor = kwargs.get("fd_inter_tensor", None)
        output_tensor = attn_metadata.output_tensor
        self.inference_ops.flash_decoding_attention(
            output_tensor,
            attn_metadata.query_states,
            attn_metadata.k_cache,
            attn_metadata.v_cache,
            attn_metadata.sequence_lengths,
            attn_metadata.block_tables,
            attn_metadata.block_size,
            attn_metadata.kv_seq_len,
            fd_inter_tensor.mid_output,
            fd_inter_tensor.exp_sums,
            fd_inter_tensor.max_logits,
            attn_metadata.alibi_slopes,
            attn_metadata.sm_scale,
        )
        return output_tensor


class TritonAttentionBackend(AttentionBackend):
    """
    Attention backend when use_cuda_kernel is False. It uses pure Triton ops for prefilling and decoding.
    """

    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        return context_attention_unpadded(
            q=attn_metadata.query_states,
            k=attn_metadata.key_states,
            v=attn_metadata.value_states,
            k_cache=attn_metadata.k_cache,
            v_cache=attn_metadata.v_cache,
            context_lengths=attn_metadata.sequence_lengths,
            block_tables=attn_metadata.block_tables,
            block_size=attn_metadata.block_size,
            output=attn_metadata.output_tensor,
            alibi_slopes=attn_metadata.alibi_slopes,
            max_seq_len=attn_metadata.kv_seq_len,
            sm_scale=attn_metadata.sm_scale,
        )

    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        fd_inter_tensor = kwargs.get("fd_inter_tensor", None)
        return flash_decoding_attention(
            q=attn_metadata.query_states,
            k_cache=attn_metadata.k_cache,
            v_cache=attn_metadata.v_cache,
            kv_seq_len=attn_metadata.sequence_lengths,
            block_tables=attn_metadata.block_tables,
            block_size=attn_metadata.block_size,
            max_seq_len_in_batch=attn_metadata.kv_seq_len,
            output=attn_metadata.output_tensor,
            mid_output=fd_inter_tensor.mid_output,
            mid_output_lse=fd_inter_tensor.mid_output_lse,
            alibi_slopes=attn_metadata.alibi_slopes,
            sm_scale=attn_metadata.sm_scale,
            kv_group_num=kwargs.get("num_key_value_groups", 1),
            q_len=kwargs.get("q_len", 1),
        )


def get_attention_backend(
    model_shard_infer_config: ModelShardInferenceConfig,
) -> AttentionBackend:
    """
    Get the attention backend based on the inference configurations. The modeling will use CUDA-kernel-based backend
    for attention module calculation only when:
        1. using CUDA kernel (use_cuda_kernel=True)
        2. can use flash attention (flash-attn installed and dtype is fp16 or bf16)
        3. not using speculative decoding (currently cuda kernel not support speculative decoding)
    Otherwise, use Triton attention backend. If found flash-attn not installed while `use_cuda_kernel` is True,
    the Triton backend will use a new k cache layout for Triton kernels.
    """
    # Currently only triton kernels support speculative decoding
    if model_shard_infer_config.use_spec_dec:
        return TritonAttentionBackend()

    if model_shard_infer_config.use_cuda_kernel:
        return CudaAttentionBackend(model_shard_infer_config.use_flash_attn)

    return TritonAttentionBackend()

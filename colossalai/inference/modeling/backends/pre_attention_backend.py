from abc import ABC, abstractmethod

from colossalai.inference.config import ModelShardInferenceConfig
from colossalai.inference.modeling.backends.attention_backend import AttentionMetaData
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import copy_k_to_blocked_cache, decoding_fused_rotary_embedding, rotary_embedding


class PreAttentionBackend(ABC):
    @abstractmethod
    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        raise NotImplementedError


class CudaPreAttentionBackend(PreAttentionBackend):
    """
    CudaPreAttentionBackend handles KV cache initialization and positional encoding for CudaAttentionBackend.
    """

    def __init__(self, use_flash_attn: bool):
        super().__init__()
        self.inference_ops = InferenceOpsLoader().load()
        self.use_flash_attn = use_flash_attn

    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        if self.use_flash_attn:
            if not attn_metadata.use_alibi_attn:
                self.inference_ops.rotary_embedding(
                    attn_metadata.query_states,
                    attn_metadata.key_states,
                    kwargs.get("cos", None),
                    kwargs.get("sin", None),
                    kwargs.get("high_precision", False),
                )
            self.inference_ops.context_kv_cache_memcpy(
                attn_metadata.key_states,
                attn_metadata.value_states,
                attn_metadata.k_cache,
                attn_metadata.v_cache,
                attn_metadata.sequence_lengths,
                attn_metadata.cu_seqlens,
                attn_metadata.block_tables,
                attn_metadata.kv_seq_len,
            )
        elif not attn_metadata.use_alibi_attn:
            rotary_embedding(
                attn_metadata.query_states,
                attn_metadata.key_states,
                kwargs.get("cos", None),
                kwargs.get("sin", None),
            )

    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        if not attn_metadata.use_alibi_attn:
            self.inference_ops.rotary_embedding_and_cache_copy(
                attn_metadata.query_states,
                attn_metadata.key_states,
                attn_metadata.value_states,
                kwargs.get("cos", None),
                kwargs.get("sin", None),
                attn_metadata.k_cache,
                attn_metadata.v_cache,
                attn_metadata.sequence_lengths,
                attn_metadata.block_tables,
                kwargs.get("high_precision", None),
            )
        else:
            self.inference_ops.decode_kv_cache_memcpy(
                attn_metadata.key_states,
                attn_metadata.value_states,
                attn_metadata.k_cache,
                attn_metadata.v_cache,
                attn_metadata.sequence_lengths,
                attn_metadata.block_tables,
            )


class TritonPreAttentionBackend(PreAttentionBackend):
    """
    TritonPreAttentionBackend handles KV cache initialization and positional encoding for TritonAttentionBackend.
    """

    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        if not attn_metadata.use_alibi_attn:
            rotary_embedding(
                attn_metadata.query_states,
                attn_metadata.key_states,
                kwargs.get("cos", None),
                kwargs.get("sin", None),
            )

    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        if not attn_metadata.use_spec_dec and not attn_metadata.use_alibi_attn:
            decoding_fused_rotary_embedding(
                attn_metadata.query_states,
                attn_metadata.key_states,
                attn_metadata.value_states,
                kwargs.get("cos", None),
                kwargs.get("sin", None),
                attn_metadata.k_cache,
                attn_metadata.v_cache,
                attn_metadata.block_tables,
                attn_metadata.sequence_lengths,
            )
        else:  # else if using speculative decoding
            if not attn_metadata.use_alibi_attn:
                rotary_embedding(
                    attn_metadata.query_states,
                    attn_metadata.key_states,
                    kwargs.get("cos", None),
                    kwargs.get("sin", None),
                )
            copy_k_to_blocked_cache(
                attn_metadata.key_states,
                attn_metadata.k_cache,
                kv_lengths=attn_metadata.sequence_lengths,
                block_tables=attn_metadata.block_tables,
                n=kwargs.get("q_len", 1),
            )
            copy_k_to_blocked_cache(
                attn_metadata.value_states,
                attn_metadata.v_cache,
                kv_lengths=attn_metadata.sequence_lengths,
                block_tables=attn_metadata.block_tables,
                n=kwargs.get("q_len", 1),
            )


def get_pre_attention_backend(
    model_shard_infer_config: ModelShardInferenceConfig,
) -> PreAttentionBackend:
    """
    Get the backend for pre-attention computations, including potisional encoding like
    RoPE and KV cache initialization. It adopt the same selection logic as attention_backend/get_attention_backend.
    """
    if model_shard_infer_config.use_spec_dec:
        return TritonPreAttentionBackend()

    if model_shard_infer_config.use_cuda_kernel:
        return CudaPreAttentionBackend(model_shard_infer_config.use_flash_attn)

    return TritonPreAttentionBackend()

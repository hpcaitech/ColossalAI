from abc import ABC, abstractmethod
import torch

from colossalai.inference.utils import can_use_flash_attn2
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.inference.modeling.backends.attention_backend import AttentionMetaData
from colossalai.logging import get_dist_logger
from colossalai.kernel.triton import (
    copy_k_to_blocked_cache,
    decoding_fused_rotary_embedding,
    rotary_embedding,
)

logger = get_dist_logger(__name__)
inference_ops = InferenceOpsLoader().load()


class PreAttentionBackend(ABC):
    @abstractmethod
    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        raise NotImplementedError


class CudaPreAttentionBackend(PreAttentionBackend):
    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        if not attn_metadata.use_alibi_attn:
            inference_ops.rotary_embedding(
                attn_metadata.query_states,
                attn_metadata.key_states,
                kwargs.get("cos", None),
                kwargs.get("sin", None),
                kwargs.get("high_precision", False),
            )
        inference_ops.context_kv_cache_memcpy(
            attn_metadata.key_states,
            attn_metadata.value_states,
            attn_metadata.k_cache,
            attn_metadata.v_cache,
            attn_metadata.sequence_lengths,
            attn_metadata.cu_seqlens,
            attn_metadata.block_tables,
            attn_metadata.kv_seq_len,
        )

    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        if not attn_metadata.use_alibi_attn:
            inference_ops.rotary_embedding_and_cache_copy(
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
            inference_ops.decode_kv_cache_memcpy(
                attn_metadata.key_states,
                attn_metadata.value_states,
                attn_metadata.k_cache,
                attn_metadata.v_cache,
                attn_metadata.sequence_lengths,
                attn_metadata.block_tables,
            )


class TritonPreAttentionBackend(PreAttentionBackend):
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
        else: # else if using speculative decoding
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
    use_spec_dec: bool, use_cuda_kernel: bool, dtype: torch.dtype
) -> PreAttentionBackend:
    """
    Get the backend for pre-attention computations, including potisional encoding like RoPE and KV cache initialization.
    """
    use_flash_attn = can_use_flash_attn2(dtype)
    if use_cuda_kernel and use_flash_attn and not use_spec_dec:
        return CudaPreAttentionBackend()
    else:
        return TritonPreAttentionBackend()

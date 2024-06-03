from abc import ABC, abstractmethod
from dataclasses import dataclass
from flash_attn import flash_attn_varlen_func
import torch

from colossalai.inference.config import InputMetaData
from colossalai.inference.utils import can_use_flash_attn2
from colossalai.logging import get_dist_logger
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import (
    context_attention_unpadded,   
    flash_decoding_attention,
)

logger = get_dist_logger(__name__)
inference_ops = InferenceOpsLoader().load()
 
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
    def prefill(self, attn_metadata: AttentionMetaData, **kwargs):
        if not attn_metadata.use_spec_dec:
            token_nums = kwargs.get('token_nums', -1)
            
            attn_output = flash_attn_varlen_func(
                attn_metadata.query_states,
                attn_metadata.key_states,
                attn_metadata.value_states,
                cu_seqlens_q=attn_metadata.cu_seqlens,
                cu_seqlens_k=attn_metadata.cu_seqlens,
                max_seqlen_k=attn_metadata.kv_seq_len,
                max_seqlen_v=attn_metadata.kv_seq_len,
                dropout_p=0.0,
                softmax_scale=attn_metadata.sm_scale,
                causal=True,
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
                max_seq_len=attn_metadata.kv_seq_len,
                sm_scale=attn_metadata.sm_scale,
                use_new_kcache_layout=True,           
            )
        return attn_output
    
                
    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        fd_inter_tensor = kwargs.get('fd_inter_tensor', None)
        output_tensor = attn_metadata.output_tensor
        inference_ops.flash_decoding_attention(
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
            max_seq_len=attn_metadata.kv_seq_len,
            sm_scale=attn_metadata.sm_scale,
            use_new_kcache_layout=False,
        )
    
    def decode(self, attn_metadata: AttentionMetaData, **kwargs):
        fd_inter_tensor = kwargs.get('fd_inter_tensor', None)
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
            sm_scale=attn_metadata.sm_scale,
            kv_group_num=kwargs.get('num_key_value_groups', 0),
            q_len=kwargs.get('q_len', 1),
        )
                  
    
def get_attention_backend(use_spec_dec: bool, use_cuda_kernel: bool, dtype: torch.dtype) -> AttentionBackend:
    use_flash_attn = can_use_flash_attn2(dtype) 
    if use_cuda_kernel and use_flash_attn and not use_spec_dec:
        return CudaAttentionBackend()
    else:
        return TritonAttentionBackend()
    
from colossalai.inference.config import InputMetaData
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer._operation import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.shardformer.shard import ShardConfig
from colossalai.kernel.triton import flash_decoding_attention_with_alibi
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.jit.bias_gelu import GeLUFunction
from colossalai.kernel.jit.bias_dropout_add import bias_dropout_add_fused_inference


import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from transformers.models.bloom.modeling_bloom import (
    BloomBlock, 
    BloomForCausalLM, 
    BloomModel,
    BloomAttention,
    BloomConfig,
    BloomMLP,
    BloomGelu,
)

from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)

inference_ops = InferenceOpsLoader().load()

try:
    from flash_attn import flash_attn_varlen_func
    
    use_flash_attn2 = True
except ImportError:
    use_flash_attn2 = False
    logger.warning(f"flash_attn2 has not been installed yet, we will use triton flash attn instead.")


# The Alibi implementation is adapted from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
def _get_alibi_slopes(n_heads: int):
    def _get_alibi_slopes_pow_of_2(n_heads):
        start = (2 ** (-2 ** -(math.log2(n_heads) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n_heads)]
    
    if math.log2(n_heads).is_integer():
        return _get_alibi_slopes_pow_of_2(n_heads)
    else: 
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return _get_alibi_slopes_pow_of_2(closest_power_of_2) + _get_alibi_slopes(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
        
def _get_alibi_tensor(n_heads: int, mask: torch.Tensor):
    slopes = _get_alibi_slopes(n_heads).to(mask.device)
    distance = mask.cumsum(dim=-1)
    return distance[:, :, None] * slopes[None, None, :]


# def _fill_with_neg_inf(t):
#     return t.float().fill_(float("-inf")).type_as(t)
       
# # (Register buffer within BloomModel), only use for inference
# def _get_alibi_tensor(max_pos: int, n_heads: int):
#     slopes = torch.Tensor(_get_alibi_slopes(n_heads))
#     alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0) \
#         .expand(n_heads, -1, -1) \
#         .view(n_heads, 1, max_pos)
        
#     alibi_mask = torch.triu (
#         _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
#     )
#     return alibi_mask.unsqueeze(0) + alibi
        

# TODO
def bloom_model_forward(
    self: BloomModel,
    input_tokens_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
    use_cuda_kernel: Optional[bool] = True,
    high_precision: bool = False,
) -> torch.Tensor:
    
    def get_alibi_mask(x: torch.Tensor, past_seq_length: int, is_prompts: bool = False):
        if is_prompts:
            is_prompts = False
            self.register_buffer("future_mask", _get_alibi_tensor())
    
    is_prompts = inputmetadata.is_prompts
    block_tables = inputmetadata.block_tables
    sequence_lengths = inputmetadata.sequence_lengths
    batch_size = inputmetadata.batch_size
    kv_seq_len = inputmetadata.kv_seq_len
    
    if batch_size >= 32 and kv_seq_len > 512:
        use_cuda_kernel = False
    
    cu_seqlens = None
    hidden_states = self.word_embeddings(input_tokens_ids)
    hidden_states = self.word_embeddings_layernorm(hidden_states)
    
    if use_cuda_kernel:
        if inputmetadata != torch.float32 and use_flash_attn2:
            cu_seqlens = F.pad(torch.cumsum(sequence_lengths, dim=0, dtype=torch.torch.int32), (1, 0))
    
    seq_length_with_past = sequence_lengths
    
    # if is_prompts:
    #     is_prompts = False
    #     self.register_buffer("future_mask", _get_alibi_tensor(self.n_head, self.max_cache_pos).to(hidden_states), persistent=False)
    # if seq_length_with_past > self.max_cache_pos:
    #     self.max_cache_pos = seq_length_with_past
    #     self.register_buffer("future_mask", _get_alibi_tensor(self.n_head, self.max_cache_pos).to(hidden_states), persistent=False)
    
    alibi = _get_alibi_slopes(self.n_head)
    # alibi_mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past]     
    
    sm_scale = 1.0 / (inputmetadata.head_dim**0.5)
    norm_output = torch.empty_like(hidden_states)
    
    for layer_id, layer in enumerate(self.h):
        hidden_states = layer(
            hidden_states,
            alibi=alibi,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            sequence_lengths=sequence_lengths,
            cu_seqlens=cu_seqlens,
            fd_inter_tensor=inputmetadata.fd_inter_tensor,
            kv_seq_len=kv_seq_len,
            output_tensor=output_tensor,
            use_cuda_kernel=use_cuda_kernel,
            high_precision=high_precision,
            norm_output=norm_output,
            sm_scale=sm_scale,
            use_cuda_kernel=use_cuda_kernel,
            high_precision=high_precision,
        )
              
    hidden_states = self.ln_f(hidden_states)
    return hidden_states    


def bloom_causal_lm_forward(
    self: BloomForCausalLM,
    input_tokens_ids: torch.Tensor,
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
) -> torch.Tensor:
    
    hidden_states = bloom_model_forward(
        self.model,
        input_tokens_ids=input_tokens_ids,
        output_tensor=output_tensor,
        inputmetadata=inputmetadata,
        k_caches=k_caches,
        v_caches=v_caches,
        use_cuda_kernel=inputmetadata.use_cuda_kernel,
        high_precision=inputmetadata.high_precision,
    )
    logits = torch.mm(hidden_states, self.lm_head.weight)
    return logits


# TODO
def bloom_block_forward(
    self: BloomBlock,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    block_tables: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    fd_inter_tensor: FDIntermTensors,
    is_prompts: bool = True,
    is_verifier: bool = False,
    tokens_to_verify: int = None,
    kv_seq_len: int = 0,
    output_tensor: torch.Tensor = None,
    norm_output: torch.Tensor = None,
    sm_scale: int = None,
    use_cuda_kernel: bool = True,
    cu_seqlens: torch.Tensor = None,
    high_precision: bool = False,
) -> torch.Tensor:
    
    # LayerNorm before attention
    layernorm_output = self.input_layernorm(hidden_states)
    
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states
    
    # Self attention
    attn_output, _ = self.self_attention(
        hidden_states=layernorm_output,
        residual=residual,
        alibi=alibi,
        hidden_states=hidden_states,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        is_verifier=is_verifier,
        tokens_to_verify=tokens_to_verify,
        sequence_lengths=sequence_lengths,
        fd_inter_tensor=fd_inter_tensor,
        kv_seq_len=kv_seq_len,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
        use_cuda_kernel=use_cuda_kernel,
        cu_seqlens=cu_seqlens,
        high_precision=high_precision,
    )
        
    # LayerNorm post attention
    layernorm_output = self.post_attention_layernorm(attn_output)
    
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = attn_output
        
    # MLP
    output = self.mlp(layernorm_output, residual) # including residuals
        
    return output
        

# TODO
class ColossalInferBloomAttention(BloomAttention):
    def __init__(
        self, 
        config: BloomConfig,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj_w: torch.Tensor = None,
    ):
        super().__init__(config)
        self.q_proj_weight = attn_qproj_w
        self.k_proj_weight = attn_kproj_w
        self.v_proj_weight = attn_vproj_w
        self.o_proj_weight = attn_oproj_w
 
        qkv_weight_list = [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]
        self.qkv_weight = torch.stack(qkv_weight_list, dim=0)

        # garbage collection
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        
    @staticmethod
    def from_native_module(module: BloomAttention, *args, **kwargs) -> BloomAttention:
        config = module.config
        attn_qproj_w = module.q_proj.weight.transpose(0, 1)
        attn_kproj_w = module.k_proj.weight.transpose(0, 1)
        attn_vproj_w = module.v_proj.weight.transpose(0, 1)
        attn_oproj_w = module.o_proj.weight.transpose(0, 1)

        attn_layer = ColossalInferBloomAttention(
            config=config,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj_w=attn_oproj_w,
        )

        return attn_layer
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        block_tables: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sequence_lengths: torch.Tensor,
        fd_inter_tensor: FDIntermTensors,
        is_prompts: bool = True,
        kv_seq_len: int = 0,
        output_tensor: torch.Tensor = None,
        sm_scale: int = None,
        use_cuda_kernel: bool = True,
        cu_seqlens: torch.Tensor = None,
        high_precision: bool = False,
    ):
        
        token_nums = hidden_states.size(0)

        hidden_states = hidden_states.expand(3, -1, -1)
        query_states, key_states, value_states = (
            torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
        )

        block_size = k_cache.size(-2)
        
        if is_prompts:  # Prefilling
            
            # TODO context stage alibi flash_attn
            pass
        
        else:   # Decoding
                            
            # If alibi in this way, then next step is to softmax with matmul_result,
            # so do I need consider how to utilize the matmul_result
            matmul_result = alibi.baddbmm(
                batch1=query_states,
                batch2=key_states,
                beta=self.beta,
                alpha=self.inv_norm_factor,
            )
                
            
            attn_output = flash_decoding_attention_with_alibi(
                q=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                alibi=alibi,
                kv_seq_len=sequence_lengths,
                block_tables=block_tables,
                block_size=block_size,
                max_seq_len_in_batch=kv_seq_len,
                output=output_tensor,
                mid_output=fd_inter_tensor.mid_output,
                mid_output_lse=fd_inter_tensor.mid_output_lse,
                sm_scale=sm_scale,
            )

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = torch.mm(attn_output, self.o_proj_weight)

        return attn_output
    

class ColossalInferBloomMLP(BloomMLP):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.gelu_impl = GeLUFunction.apply
        
    @staticmethod
    def from_native_method(module: BloomMLP, *args, **kwargs) -> BloomMLP:
        config = module.config
        mlp_layer = ColossalInferBloomMLP(config=config)
        return mlp_layer
    
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        bias = torch.zero_like(hidden_states)
        hidden_states = self.gelu_impl(hidden_states, bias)
        intermediate_output = self.dense_4h_to_h(hidden_states)
        output = bias_dropout_add_fused_inference(intermediate_output, bias, residual, self.hidden_dropout)
        return output
        
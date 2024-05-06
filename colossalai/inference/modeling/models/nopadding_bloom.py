from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomModel

from colossalai.inference.config import InputMetaData
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.utils import get_alibi_slopes
from colossalai.kernel.jit.bias_dropout_add import bias_dropout_add_fused_inference
from colossalai.kernel.jit.bias_gelu import GeLUFunction
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import context_attention_unpadded, copy_k_to_blocked_cache, flash_decoding_attention
from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)

try:
    from flash_attn import flash_attn_varlen_func

    use_flash_attn2 = True
except ImportError:
    use_flash_attn2 = False
    logger.warning(f"flash_attn2 has not been installed yet, we will use triton flash attn instead.")

inference_ops = InferenceOpsLoader().load()

logger = get_dist_logger(__name__)


def bloom_causal_lm_forward(
    self: BloomForCausalLM,
    input_tokens_ids: torch.Tensor,  # no padding
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Replacement of forward function in BloomForCausalLM.

    Args:
        input_tokens_ids (torch.Tensor): Input token Ids with no paddings.
        output_tensor (torch.Tensor): Intermediate tensor to hold attention output.
        inputmetadata (InputMetaData): Ths input metadata for a single step.
        k_caches (List[torch.Tensor], optional): List of key caches. Defaults to None.
        v_caches (List[torch.Tensor], optional): List of value caches. Defaults to None.

    Returns:
        torch.Tensor: Logits.
    """
    # print(f"[BloomForCausalLM] input input_tokens_ids {input_tokens_ids}")

    hidden_states = bloom_model_forward(
        self.transformer,
        input_tokens_ids=input_tokens_ids,
        output_tensor=output_tensor,
        inputmetadata=inputmetadata,
        k_caches=k_caches,
        v_caches=v_caches,
        use_cuda_kernel=inputmetadata.use_cuda_kernel,
        high_precision=inputmetadata.high_precision,
    )

    logits = self.lm_head(hidden_states)
    # print(f"[BloomForCausalLM] output logits {logits}")
    return logits


def bloom_model_forward(
    self: BloomModel,
    input_tokens_ids: torch.Tensor,  # no padding
    output_tensor: torch.Tensor,
    inputmetadata: InputMetaData,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
    use_cuda_kernel: Optional[bool] = True,
    high_precision: bool = False,
) -> torch.Tensor:
    """
    Replacement of forward function in BloomModel.

    Args:
        input_tokens_ids (torch.Tensor): Input token IDs with no padding.
        output_tensor (torch.Tensor): Intermediate tensor to hold attention output.
        inputmetadata (InputMetaData): Ths input metadata for a single step.
        k_caches (List[torch.Tensor], optional): List of k caches. Defaults to None.
        v_caches (List[torch.Tensor], optional): List of v caches. Defaults to None.
        use_cuda_kernel (Optional[bool], optional): Whether to use CUDA kernel. Defaults to True.
        high_precision (bool, optional): Whether to use high precision. Defaults to False.

    Returns:
        torch.Tensor: Hidden states.
    """
    # print(f"[BloomModel] input_tokens_ids {input_tokens_ids}")

    block_tables = inputmetadata.block_tables
    sequence_lengths = inputmetadata.sequence_lengths
    batch_size = inputmetadata.batch_size
    kv_seq_len = inputmetadata.kv_seq_len

    if batch_size >= 32 and kv_seq_len > 512:
        use_cuda_kernel = False

    cu_seqlens = None

    if use_cuda_kernel:
        if inputmetadata.dtype != torch.float32 and use_flash_attn2:
            cu_seqlens = F.pad(torch.cumsum(sequence_lengths, dim=0, dtype=torch.torch.int32), (1, 0))

    input_embeds = self.word_embeddings(input_tokens_ids)
    hidden_states = self.word_embeddings_layernorm(input_embeds)

    sm_scale = 1.0 / (inputmetadata.head_dim**0.5)
    norm_output = torch.empty_like(hidden_states)

    for layer_id, layer in enumerate(self.h):
        hidden_states = layer(
            hidden_states,
            block_tables=block_tables,
            is_prompts=inputmetadata.is_prompts,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            sequence_lengths=sequence_lengths,
            cu_seqlens=cu_seqlens,
            fd_inter_tensor=inputmetadata.fd_inter_tensor,
            kv_seq_len=kv_seq_len,
            output_tensor=output_tensor,
            norm_output=norm_output,
            sm_scale=sm_scale,
            use_cuda_kernel=use_cuda_kernel,
            high_precision=high_precision,
        )

    # print(f"[BloomModel] hidden_states output before cumsum {hidden_states}")

    if inputmetadata.is_prompts:
        seq_len_cumsum = sequence_lengths.cumsum(dim=0)
        hidden_states = hidden_states[seq_len_cumsum - 1].contiguous()

    hidden_states = self.ln_f(hidden_states)

    # print(f"[BloomModel] hidden_states output {hidden_states}")
    return hidden_states


def bloom_block_forward(
    self: BloomBlock,
    hidden_states: torch.Tensor,
    block_tables: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    fd_inter_tensor: FDIntermTensors,
    is_prompts: bool = True,
    kv_seq_len: int = 0,
    output_tensor: torch.Tensor = None,
    norm_output: torch.Tensor = None,
    sm_scale: int = None,
    use_cuda_kernel: bool = True,
    cu_seqlens: torch.Tensor = None,
    high_precision: bool = False,
) -> torch.FloatTensor:
    """
    Replacement of forward function in the BloomBlock module.

    Args:
        hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
            storing mapping of token_position_id -> block_id.
        k_cache (torch.Tensor): It holds the GPU memory for the key cache.
        v_cache (torch.Tensor): It holds the GPU memory for the key cache.
        sequence_lengths (torch.Tensor): Holding the sequence length of each sequence.
        fd_inter_tensor (FDIntermTensors): Holding tensors used for
            storing intermediate values in flash-decoding.
        is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
        kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
        output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
        norm_output (torch.Tensor, optional): The mid tensor holds the output of layernorm. Defaults to None.
        sm_scale (int, optional): Used for flash attention. Defaults to None.
        use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
        cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
        high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.

    Returns:
        torch.Tensor: The output tensor.
    """

    # print(f"[BloomBlock] input hidden_states {hidden_states}")

    # LayerNorm before attention
    norm_output = self.input_layernorm(hidden_states)

    if self.apply_residual_connection_post_layernorm:
        residual = norm_output
    else:
        residual = hidden_states

    # Self attention
    attn_outputs = self.self_attention(
        hidden_states=norm_output,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        sequence_lengths=sequence_lengths,
        fd_inter_tensor=fd_inter_tensor,
        kv_seq_len=kv_seq_len,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
        use_cuda_kernel=use_cuda_kernel,
        cu_seqlens=cu_seqlens,
        high_precision=high_precision,
    )

    # attention_output = attn_outputs[0]
    # outputs = attn_outputs[1:]
    attention_output = attn_outputs + residual

    # LayerNorm post attention
    norm_output = self.post_attention_layernorm(attention_output)

    if self.apply_residual_connection_post_layernorm:
        residual = norm_output
    else:
        residual = attention_output

    # MLP (including residuals)
    output = self.mlp(norm_output, residual)

    # print(f"[DEBUG] output shape {output.shape}, and outputs shape {outputs.shape}")
    # print(f"[DEBUG] output type {output.dtype}, and outputs type {outputs.dtype}")
    # outputs = output + outputs

    # return outputs

    # print(f"[BloomBlock] output {output}")
    return output


# class NopadBloomAttention(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         n_heads: int,
#         attn_qproj_w: torch.Tensor = None,
#         attn_kproj_w: torch.Tensor = None,
#         attn_vproj_w: torch.Tensor = None,
#         attn_oproj_w: torch.Tensor = None,
#     ):
#         """
#         Customized attention layer for Bloom model.

#         Args:
#             hidden_size (int): Imensionality of the embeddings and hidden states.
#             n_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
#             attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
#             attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
#             attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
#             attn_oproj_w (torch.Tensor, optional): The transposed o_proj weight. Defaults to None.
#         """
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.num_heads = n_heads
#         self.alibi_slopes = get_alibi_slopes(self.num_heads, device=attn_qproj_w.device)
#         self.head_dim = self.hidden_size // self.num_heads
#         self.dense = attn_oproj_w

#         qkv_weight_list = [attn_qproj_w, attn_kproj_w, attn_vproj_w]
#         self.qkv_weight = torch.stack(qkv_weight_list, dim=0)

#     @staticmethod
#     def from_native_module(module: nn.Module, *args, **kwargs) -> "NopadBloomAttention":
#         """
#         Initialize the weight of NopadBloomAttention from the original BloomAttention.

#         Args:
#             module (nn.Module): The original BloomAttention layer.

#         Returns:
#             NopadBloomAttention: The initialized NopadBloomAttention layer.
#         """

#         hidden_size = module.hidden_size
#         num_heads = module.num_heads
#         q_proj_w, k_proj_w, v_proj_w = module.query_key_value.weight.view((3, hidden_size, hidden_size))

#         attn_qproj_w = q_proj_w.transpose(0, 1)
#         attn_kproj_w = k_proj_w.transpose(0, 1)
#         attn_vproj_w = v_proj_w.transpose(0, 1)
#         attn_oproj_w = module.dense.weight.transpose(0, 1)

#         attn_layer = NopadBloomAttention(
#             hidden_size=hidden_size,
#             n_heads=num_heads,
#             attn_qproj_w=attn_qproj_w,
#             attn_kproj_w=attn_kproj_w,
#             attn_vproj_w=attn_vproj_w,
#             attn_oproj_w=attn_oproj_w,
#         )

#         return attn_layer

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         block_tables: torch.Tensor,
#         k_cache: torch.Tensor,
#         v_cache: torch.Tensor,
#         sequence_lengths: torch.Tensor,
#         fd_inter_tensor: FDIntermTensors,
#         is_prompts: bool = True,
#         kv_seq_len: int = 0,
#         output_tensor: torch.Tensor = None,
#         sm_scale: int = None,
#         use_cuda_kernel: bool = True,
#         cu_seqlens: torch.Tensor = None,
#         high_precision: bool = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         """
#         Forward function of the NopadBloomAttention. Current attention does not support speculative decoding.

#         Args:
#             hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
#             block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
#                 storing mapping of token_position_id -> block_id.
#             k_cache (torch.Tensor): It holds the GPU memory for the key cache.
#             v_cache (torch.Tensor): It holds the GPU memory for the key cache.
#             sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence.
#             cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin.
#             fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
#                 storing intermediate values in flash-decoding.
#             is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
#             kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
#             output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
#             sm_scale (int, optional): Used for flash attention. Defaults to None.
#             use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
#             cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
#             high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
#         """

#         print(f"[BloomAttention] input hidden_states {hidden_states}")
#         token_nums = hidden_states.size(0)
#         hidden_states = hidden_states.expand(3, -1, -1)
#         query_states, key_states, value_states = (
#             torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
#         )

#         block_size = k_cache.size(-2)

#         if is_prompts:  # Context stage (prefilling phase)
#             if (
#                 use_cuda_kernel
#                 and query_states.dtype != torch.float32
#                 and use_flash_attn2  # flash attn 2 currently only supports FP16/BF16
#             ):
#                 # Copy the GPU memory of kvcache during context stage
#                 inference_ops.context_kv_cache_memcpy(
#                     key_states, value_states, k_cache, v_cache, sequence_lengths, cu_seqlens, block_tables, kv_seq_len
#                 )

#                 attn_output = flash_attn_varlen_func(
#                     query_states,
#                     key_states,
#                     value_states,
#                     cu_seqlens_q=cu_seqlens,
#                     cu_seqlens_k=cu_seqlens,
#                     max_seqlen_q=kv_seq_len,
#                     max_seqlen_k=kv_seq_len,
#                     dropout_p=0.0,
#                     softmax_scale=sm_scale,
#                     causal=True,
#                     alibi_slopes=self.alibi_slopes,
#                 )
#                 attn_output = attn_output.view(token_nums, -1)

#             else:
#                 attn_output = context_attention_unpadded(
#                     q=query_states,
#                     k=key_states,
#                     v=value_states,
#                     k_cache=k_cache,
#                     v_cache=v_cache,
#                     context_lengths=sequence_lengths,
#                     block_size=block_size,
#                     block_tables=block_tables,
#                     output=output_tensor,
#                     alibi_slopes=self.alibi_slopes,
#                     max_seq_len=kv_seq_len,
#                     sm_scale=sm_scale,
#                 )

#         else:  # Decode stage
#             if use_cuda_kernel:
#                 # Copy the GPU memory of kvcache during decode stage
#                 inference_ops.decode_kv_cache_memcpy(
#                     key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables
#                 )
#             else:
#                 copy_k_to_blocked_cache(
#                     key_states,
#                     k_cache,
#                     kv_lengths=sequence_lengths,
#                     block_tables=block_tables,
#                 )
#                 copy_k_to_blocked_cache(
#                     value_states,
#                     v_cache,
#                     kv_lengths=sequence_lengths,
#                     block_tables=block_tables,
#                 )

#             attn_output = flash_decoding_attention(
#                 q=query_states,
#                 k_cache=k_cache,
#                 v_cache=v_cache,
#                 alibi_slopes=self.alibi_slopes,
#                 kv_seq_len=sequence_lengths,
#                 block_tables=block_tables,
#                 block_size=block_size,
#                 max_seq_len_in_batch=kv_seq_len,
#                 output=output_tensor,
#                 mid_output=fd_inter_tensor.mid_output,
#                 mid_output_lse=fd_inter_tensor.mid_output_lse,
#                 sm_scale=sm_scale,
#             )

#         attn_output = attn_output.view(-1, self.hidden_size)
#         attn_output = torch.mm(attn_output, self.dense)
#         print(f"[BloomAttention] output attn_output {attn_output}")
#         return attn_output


class NopadBloomMLP(nn.Module):
    def __init__(self, hidden_size: int, hidden_dropout: float = 0.0):
        """
        Customized MLP layer for the BloomModel to replace BloomMLP.

        Args:
            hidden_size (int): The size of the hidden layer.
            hidden_dropout (float, optional): The dropout rate for the hidden layer. Defaults to 0.0.
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.dense_h_to_4h = nn.Linear(hidden_size, hidden_size * 4)
        self.gelu_impl = GeLUFunction.apply
        self.dense_4h_to_h = nn.Linear(hidden_size * 4, hidden_size)

        # self.dense_h_to_4h = self.dense_h_to_4h.half()
        # self.dense_4h_to_h = self.dense_4h_to_h.half()

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> "NopadBloomMLP":
        """
        Initialize the weight of NopadBloomMLP from original BloomMLP.

        Args:
            module (nn.Module): The original BloomMLP layer.

        Returns:
            NopadBloomMLP: The initialized NopadBloomMLP layer.
        """
        hidden_size = module.dense_h_to_4h.weight.size(1)
        mlp_layer = NopadBloomMLP(hidden_size=hidden_size, hidden_dropout=module.hidden_dropout)
        return mlp_layer

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward function of NopafBloomMLP.

        Args:
            hidden_states (torch.Tensor): The input tensor with shape [token_num, embed_dim].
            residual (torch.Tensor): The residual tensor with shape [token_num, embed_dim].

        Returns:
            torch.Tensor: The output tensor with shape [token_num, embed_dim].
        """

        # print(f"[BloomMLP] intput hidden_states {hidden_states}")
        hidden_states = self.dense_h_to_4h(hidden_states)
        bias = torch.zeros_like(hidden_states)
        hidden_states = self.gelu_impl(hidden_states, bias)
        intermediate_output = self.dense_4h_to_h(hidden_states)
        bias = torch.zeros_like(intermediate_output)
        output = bias_dropout_add_fused_inference(intermediate_output, bias, residual, self.hidden_dropout)

        # print(f"[BloomMLP] output {output}")
        return output


class NopadBloomAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj_w: torch.Tensor = None,
    ):
        """
        Customized attention layer for Bloom model.

        Args:
            hidden_size (int): Imensionality of the embeddings and hidden states.
            n_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj_w (torch.Tensor, optional): The transposed o_proj weight. Defaults to None.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = n_heads
        self.alibi_slopes = get_alibi_slopes(self.num_heads, device=attn_qproj_w.device)
        self.head_dim = self.hidden_size // self.num_heads
        self.o_proj_weight = attn_oproj_w

        qkv_weight_list = [attn_qproj_w, attn_kproj_w, attn_vproj_w]
        self.qkv_weight = torch.stack(qkv_weight_list, dim=0)  # Multi Head Attention fusion
        # print(f"[DEBUG] qkv_weight {self.qkv_weight}")

    @staticmethod
    def from_native_module(module: BloomAttention, *args, **kwargs) -> "NopadBloomAttention":
        """
        Initialize the weight of NopadBloomAttention from the original BloomAttention.

        Args:
            module (BloomAttention): The original BloomAttention layer.

        Returns:
            NopadBloomAttention: The initialized NopadBloomAttention layer.
        """

        hidden_size = module.hidden_size
        num_heads = module.num_heads
        q_proj_w, k_proj_w, v_proj_w = module.query_key_value.weight.view((3, hidden_size, hidden_size))

        # print(f"[DEBUG] original query_key_value weight {module.query_key_value.weight},\n q_proj_w {q_proj_w}, \n k_proj_w {k_proj_w}, \n v_proj_w {v_proj_w}")

        attn_qproj_w = q_proj_w.transpose(0, 1)
        attn_kproj_w = k_proj_w.transpose(0, 1)
        attn_vproj_w = v_proj_w.transpose(0, 1)
        attn_oproj_w = module.dense.weight.transpose(0, 1)

        attn_layer = NopadBloomAttention(
            hidden_size=hidden_size,
            n_heads=num_heads,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj_w=attn_oproj_w,
        )
        return attn_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward function of the NopadBloomAttention. Current attention does not support speculative decoding.

        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
            block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
                storing mapping of token_position_id -> block_id.
            k_cache (torch.Tensor): It holds the GPU memory for the key cache.
            v_cache (torch.Tensor): It holds the GPU memory for the key cache.
            sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence.
            cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin.
            fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
                storing intermediate values in flash-decoding.
            is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
            kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
            output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
            sm_scale (int, optional): Used for flash attention. Defaults to None.
            use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
            cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
            high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
        """

        print(f"[BloomAttention] input hidden_states {hidden_states}")
        token_nums = hidden_states.size(0)
        hidden_states = hidden_states.expand(3, -1, -1)
        query_states, key_states, value_states = (
            torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
        )

        # fused_qkv = torch.bmm(hidden_states, self.qkv_weight)
        # print(f"[TEST] hidden_state {hidden_states} with shape {hidden_states.shape}\n qkv_weight {self.qkv_weight} with shape {self.qkv_weight.shape}")

        # print(f"[DEBUG] after qkv: query_states {query_states} with shape {query_states.shape}, \nkey_states {key_states},\n value_states {value_states}")
        block_size = k_cache.size(-2)

        if is_prompts:  # Context stage (prefilling phase)
            if (
                use_cuda_kernel
                and query_states.dtype != torch.float32
                and use_flash_attn2  # flash attn 2 currently only supports FP16/BF16
            ):
                # Copy the GPU memory of kvcache during context stage
                inference_ops.context_kv_cache_memcpy(
                    key_states, value_states, k_cache, v_cache, sequence_lengths, cu_seqlens, block_tables, kv_seq_len
                )

                attn_output = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=kv_seq_len,
                    max_seqlen_k=kv_seq_len,
                    dropout_p=0.0,
                    softmax_scale=sm_scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                )
                attn_output = attn_output.view(token_nums, -1)

            else:
                attn_output = context_attention_unpadded(
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    context_lengths=sequence_lengths,
                    block_size=block_size,
                    block_tables=block_tables,
                    output=output_tensor,
                    alibi_slopes=self.alibi_slopes,
                    max_seq_len=kv_seq_len,
                    sm_scale=sm_scale,
                )

        else:  # Decode stage
            if use_cuda_kernel:
                # Copy the GPU memory of kvcache during decode stage
                inference_ops.decode_kv_cache_memcpy(
                    key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables
                )
            else:
                copy_k_to_blocked_cache(
                    key_states,
                    k_cache,
                    kv_lengths=sequence_lengths,
                    block_tables=block_tables,
                )
                copy_k_to_blocked_cache(
                    value_states,
                    v_cache,
                    kv_lengths=sequence_lengths,
                    block_tables=block_tables,
                )

            attn_output = flash_decoding_attention(
                q=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                alibi_slopes=self.alibi_slopes,
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
        # print(f"[BloomAttention] output attn_output {attn_output}")
        return attn_output


def bloom_attention_forward(
    self: BloomAttention,
    hidden_states: torch.Tensor,
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
    # print(f"[BloomAttention] input hidden_states {hidden_states}")
    alibi_slopes = get_alibi_slopes(self.num_heads, device=self.query_key_value.weight.device)
    token_nums = hidden_states.size(0)
    block_size = k_cache.size(-2)

    fused_qkv = self.query_key_value(hidden_states.unsqueeze(0))
    (query_states, key_states, value_states) = self._split_heads(fused_qkv)  # [bsz, seq_len, num_heads, head_dim

    # print(f"[TEST] before merge bsz, query_states {query_states} with shape {query_states.shape}, \nkey_states {key_states},\n value_states {value_states}")

    # [bsz * seq_len, num_heads head_dim]
    query_states = query_states.view(-1, self.num_heads, self.head_dim)
    key_states = key_states.view(-1, self.num_heads, self.head_dim)
    value_states = value_states.view(-1, self.num_heads, self.head_dim)

    if is_prompts:  # Context stage (prefilling phase)
        if (
            use_cuda_kernel
            and query_states.dtype != torch.float32
            and use_flash_attn2  # flash attn 2 currently only supports FP16/BF16
        ):
            # Copy the GPU memory of kvcache during context stage
            inference_ops.context_kv_cache_memcpy(
                key_states, value_states, k_cache, v_cache, sequence_lengths, cu_seqlens, block_tables, kv_seq_len
            )

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=kv_seq_len,
                max_seqlen_k=kv_seq_len,
                dropout_p=0.0,
                softmax_scale=sm_scale,
                causal=True,
                alibi_slopes=alibi_slopes,
            )
            attn_output = attn_output.view(token_nums, -1)

        else:
            attn_output = context_attention_unpadded(
                q=query_states,
                k=key_states,
                v=value_states,
                k_cache=k_cache,
                v_cache=v_cache,
                context_lengths=sequence_lengths,
                block_size=block_size,
                block_tables=block_tables,
                output=output_tensor,
                alibi_slopes=alibi_slopes,
                max_seq_len=kv_seq_len,
                sm_scale=sm_scale,
            )

    else:  # Decode stage
        if use_cuda_kernel:
            # Copy the GPU memory of kvcache during decode stage
            inference_ops.decode_kv_cache_memcpy(
                key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables
            )
        else:
            copy_k_to_blocked_cache(
                key_states,
                k_cache,
                kv_lengths=sequence_lengths,
                block_tables=block_tables,
            )
            copy_k_to_blocked_cache(
                value_states,
                v_cache,
                kv_lengths=sequence_lengths,
                block_tables=block_tables,
            )

        attn_output = flash_decoding_attention(
            q=query_states,
            k_cache=k_cache,
            v_cache=v_cache,
            alibi_slopes=alibi_slopes,
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
    attn_output = self.dense(attn_output)
    # print(f"[BloomAttention] output attn_output {attn_output}")
    return attn_output

# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
from typing import List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)

from colossalai.inference.batch_bucket import BatchBucket
from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.kernel.triton import (
    context_attention_unpadded,
    copy_k_to_blocked_cache,
    decoding_fused_rotary_embedding,
    flash_decoding_attention,
    get_xine_cache,
    rotary_embedding,
)
from colossalai.logging import get_dist_logger

logger = get_dist_logger(__name__)

try:
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning(f"triton has not been installed yet, we will use torch to complete the attention calculation.")


def llama_causal_lm_forward(
    self: LlamaForCausalLM,
    batch: BatchBucket = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    """This function will replace the forward function of LlamaForCausalLM.

    Args:
        batch (BatchInfo, optional): It stores the necessary input information for this inference. Defaults to None.
        k_caches (List[torch.Tensor], optional): It holds the GPU memory for the key cache. Defaults to None.
        v_caches (List[torch.Tensor], optional): It holds the GPU memory for the value cache. Defaults to None.
    """

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    hidden_states = llama_model_forward(
        self.model,
        batch=batch,
        k_caches=k_caches,
        v_caches=v_caches,
    )
    logits = torch.mm(hidden_states, self.lm_head.weight)
    return logits


def llama_model_forward(
    self: LlamaModel,
    batch: BatchBucket = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    """This function will replace the forward function of LlamaModel.

    Args:
        batch (BatchInfo, optional): It stores the necessary input information for this inference.. Defaults to None.
        k_caches (List[torch.Tensor], optional): It holds the GPU memory for the key cache. Defaults to None.
        v_caches (List[torch.Tensor], optional): It holds the GPU memory for the value cache. Defaults to None.
    """
    input_ids = batch.get_1D_inputs()
    block_tables = batch.get_block_table_tensor()
    sequence_lengths = batch.get_sequence_lengths()
    batch_size = batch.current_batch_size
    kv_seq_len = sequence_lengths.max().item()

    hidden_states = self.embed_tokens(input_ids)

    if batch.use_spec_dec:
        # For speculative-decoding Prefill and Verifying Stage
        if batch.is_prompts:
            # output tensor shape is the same as normal Prefill Stage
            o_tensor_size = (sequence_lengths.sum().item(), batch.num_heads * batch.head_dim)
            rotary_indexes = [torch.arange(0, length) for length in sequence_lengths]
        else:
            # the number of tokens to be verified in parallel plus the correct token in the last step
            n_tokens = batch.num_tokens_to_verify + 1
            assert n_tokens == hidden_states.size(0)
            o_tensor_size = (batch_size * n_tokens, batch.num_heads * batch.head_dim)
            rotary_indexes = [(length - n_tokens + i).view(-1) for i in range(n_tokens) for length in sequence_lengths]
        rotary_indexes = torch.cat(rotary_indexes, dim=-1)
        cos_sin = (self._cos_cached[rotary_indexes], self._sin_cached[rotary_indexes])
    else:
        # For normal Prefill and Decoding Stage
        if batch.is_prompts:
            o_tensor_size = (sequence_lengths.sum().item(), batch.num_heads * batch.head_dim)
        else:
            o_tensor_size = (batch_size, batch.num_heads * batch.head_dim)
        cos_sin = get_xine_cache(sequence_lengths, self._cos_cached, self._sin_cached, batch.is_prompts)

    sm_scale = 1.0 / (batch.head_dim**0.5)
    output_tensor = torch.zeros(o_tensor_size, dtype=batch.dtype, device=batch.device)
    norm_output = torch.empty_like(hidden_states)
    tokens_to_verify = batch.num_tokens_to_verify if batch.use_spec_dec else None
    residual = None

    for layer_id, decoder_layer in enumerate(self.layers):
        hidden_states, residual = decoder_layer(
            hidden_states,
            residual=residual,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=batch.is_prompts,
            is_verifier=batch.use_spec_dec,
            tokens_to_verify=tokens_to_verify,
            sequence_lengths=sequence_lengths,
            kv_seq_len=kv_seq_len,
            cos_sin=cos_sin,
            fd_inter_tensor=batch.fd_inter_tensor,
            output_tensor=output_tensor,
            norm_output=norm_output,
            sm_scale=sm_scale,
        )

    if batch.is_prompts:
        seq_len_cumsum = sequence_lengths.cumsum(dim=0)
        hidden_states = hidden_states[seq_len_cumsum - 1].contiguous()
        residual = residual[seq_len_cumsum - 1].contiguous()
        norm_output = torch.empty_like(hidden_states)
    hidden_states, _ = self.norm(hidden_states, norm_output, residual)

    return hidden_states


def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    is_verifier: bool = False,
    tokens_to_verify: int = None,
    sequence_lengths: torch.Tensor = None,
    kv_seq_len: int = 0,
    cos_sin: Tuple[torch.Tensor] = None,
    fd_inter_tensor: FDIntermTensors = None,
    output_tensor: torch.Tensor = None,
    norm_output: torch.Tensor = None,
    sm_scale: int = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """This function will replace the forward function of LlamaDecoderLayer.

    Args:
        hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        residual (torch.Tensor): shape [token_num, embed_dim], used to be added to hidden_states in out_proj.
        block_tables (torch.Tensor, optional): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
            storing mapping of token_position_id -> block_id. Defaults to None.
        k_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
        v_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
        is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
        sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence. Defaults to None.
        kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
        cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin. Defaults to None.
        fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
            storing intermediate values in flash-decoding. Defaults to None.
        output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
        norm_output (torch.Tensor, optional): The mid tensor holds the output of layernorm. Defaults to None.
        sm_scale (int, optional): Used for flash attention. Defaults to None.
    """

    hidden_states, residual = self.input_layernorm(hidden_states, norm_output, residual)
    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        is_verifier=is_verifier,
        tokens_to_verify=tokens_to_verify,
        sequence_lengths=sequence_lengths,
        kv_seq_len=kv_seq_len,
        cos_sin=cos_sin,
        fd_inter_tensor=fd_inter_tensor,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, norm_output, residual)
    hidden_states = self.mlp(hidden_states)

    return hidden_states, residual


class NopadLlamaAttention(LlamaAttention):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj_w: torch.Tensor = None,
    ):
        """This layer will replace the LlamaAttention.

        Args:
            config (LlamaConfig): Holding the Llama model config.
            layer_idx (Optional[int], optional): The decode layer id of this attention layer. Defaults to None.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj_w (torch.Tensor, optional): The transposed o_proj weight. Defaults to None.
        """
        super().__init__(config, layer_idx)
        self.q_proj_weight = attn_qproj_w
        self.k_proj_weight = attn_kproj_w
        self.v_proj_weight = attn_vproj_w
        self.o_proj_weight = attn_oproj_w

        if self.num_heads == self.num_key_value_heads:
            qkv_weight_list = [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight]
            self.qkv_weight = torch.stack(qkv_weight_list, dim=0)

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    @staticmethod
    def from_native_module(module: LlamaAttention, *args, **kwargs) -> LlamaAttention:
        """Used for initialize the weight of NopadLlamaAttention by origin LlamaAttention.

        Args:
            module (LlamaAttention): The origin LlamaAttention layer.
        """
        config = module.config
        layer_idx = module.layer_idx

        attn_qproj_w = module.q_proj.weight.transpose(0, 1)
        attn_kproj_w = module.k_proj.weight.transpose(0, 1)
        attn_vproj_w = module.v_proj.weight.transpose(0, 1)
        attn_oproj_w = module.o_proj.weight.transpose(0, 1)

        attn_layer = NopadLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj_w=attn_oproj_w,
        )

        return attn_layer

    # Replace transformers.models.llama.modeling_llama.LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_tables: torch.Tensor = None,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        is_prompts: bool = True,
        is_verifier: bool = False,
        tokens_to_verify: int = None,
        sequence_lengths: torch.Tensor = None,
        kv_seq_len: int = 0,
        cos_sin: Tuple[torch.Tensor] = None,
        fd_inter_tensor: FDIntermTensors = None,
        output_tensor: torch.Tensor = None,
        sm_scale: int = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
            block_tables (torch.Tensor, optional): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
                storing mapping of token_position_id -> block_id. Defaults to None.
            k_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
            v_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
            is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
            sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence. Defaults to None.
            kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
            cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin. Defaults to None.
            fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
                storing intermediate values in flash-decoding. Defaults to None.
            output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
            sm_scale (int, optional): Used for flash attention. Defaults to None.
        """

        if self.num_heads != self.num_key_value_heads:
            query_states = torch.mm(hidden_states, self.q_proj_weight).view(-1, self.num_heads, self.head_dim)
            key_states = torch.mm(hidden_states, self.k_proj_weight).view(-1, self.num_key_value_heads, self.head_dim)
            value_states = torch.mm(hidden_states, self.v_proj_weight).view(-1, self.num_key_value_heads, self.head_dim)
        else:
            # fused qkv
            token_nums = hidden_states.size(0)
            hidden_states = hidden_states.expand(3, -1, -1)
            query_states, key_states, value_states = (
                torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
            )

        block_size = k_cache.size(-2)

        if is_prompts:
            rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])
            attn_output = context_attention_unpadded(
                q=query_states,
                k=key_states,
                v=value_states,
                k_cache=k_cache,
                v_cache=v_cache,
                context_lengths=sequence_lengths,
                block_tables=block_tables,
                block_size=block_size,
                output=output_tensor,
                max_seq_len=kv_seq_len,
                sm_scale=sm_scale,
            )
        else:
            q_len = tokens_to_verify + 1 if is_verifier else 1

            if is_verifier:
                rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])
                copy_k_to_blocked_cache(
                    key_states, k_cache, kv_lengths=sequence_lengths, block_tables=block_tables, n=q_len
                )
                copy_k_to_blocked_cache(
                    value_states, v_cache, kv_lengths=sequence_lengths, block_tables=block_tables, n=q_len
                )
            else:
                decoding_fused_rotary_embedding(
                    query_states,
                    key_states,
                    value_states,
                    cos_sin[0],
                    cos_sin[1],
                    k_cache,
                    v_cache,
                    block_tables,
                    sequence_lengths,
                )

            attn_output = flash_decoding_attention(
                q=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                kv_seq_len=sequence_lengths,
                block_tables=block_tables,
                block_size=block_size,
                max_seq_len_in_batch=kv_seq_len,
                output=output_tensor,
                mid_output=fd_inter_tensor.mid_output,
                mid_output_lse=fd_inter_tensor.mid_output_lse,
                sm_scale=sm_scale,
                q_len=q_len,
            )

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = torch.mm(attn_output, self.o_proj_weight)

        return attn_output


# NOTE This will cause the result to be different from the transformer in some cases.
class NopadLlamaMLP(LlamaMLP):
    def __init__(
        self,
        config: LlamaConfig,
        mlp_gproj_w: torch.Tensor = None,
        mlp_uproj_w: torch.Tensor = None,
        mlp_dproj_w: torch.Tensor = None,
    ):
        """This layer will replace the LlamaAttention.

        Args:
            config (LlamaConfig): Holding the Llama model config.
            mlp_gproj_w (torch.Tensor, optional): The transposed gate_proj weight. Defaults to None.
            mlp_uproj_w (torch.Tensor, optional): The transposed up_proj weight. Defaults to None.
            mlp_dproj_w (torch.Tensor, optional): The transposed down_proj weight. Defaults to None.
        """
        super().__init__(config)
        self.gate_up_weight = torch.stack([mlp_gproj_w, mlp_uproj_w], dim=0)
        self.down_proj_weight = mlp_dproj_w
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    @staticmethod
    def from_native_module(module: LlamaMLP, *args, **kwargs) -> LlamaMLP:
        """Used for initialize the weight of NopadLlamaMLP by origin LlamaMLP.

        Args:
            module (LlamaMLP): The origin LlamaMLP layer.
        """
        config = module.config

        mlp_gproj_w = module.gate_proj.weight.transpose(0, 1)
        mlp_uproj_w = module.up_proj.weight.transpose(0, 1)
        mlp_dproj_w = module.down_proj.weight.transpose(0, 1)

        mlp_layer = NopadLlamaMLP(
            config=config,
            mlp_gproj_w=mlp_gproj_w,
            mlp_uproj_w=mlp_uproj_w,
            mlp_dproj_w=mlp_dproj_w,
        )

        return mlp_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        """
        hidden_states = hidden_states.expand(2, -1, -1)
        gate_up_proj_out = torch.bmm(hidden_states, self.gate_up_weight)
        act_out = torch.nn.functional.silu(gate_up_proj_out[0], inplace=True)
        tmp_out = act_out * gate_up_proj_out[1]
        return torch.mm(tmp_out, self.down_proj_weight)

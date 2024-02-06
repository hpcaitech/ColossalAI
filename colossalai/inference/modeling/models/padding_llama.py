# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
from typing import List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)

from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.layers.attention import PagedAttention
from colossalai.inference.struct import BatchInfo
from colossalai.kernel.triton import (
    context_attention_unpadded,
    copy_kv_to_blocked_cache,
    flash_decoding_attention,
    get_xine_cache,
    rotary_embedding,
)
from colossalai.logging import get_dist_logger

from flash_attn.bert_padding import index_first_axis, pad_input  # noqa

logger = get_dist_logger(__name__)

try:
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning(f"triton has not been installed yet, we will use torch to complete the attention calculation.")


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def llama_causal_lm_forward(
    self: LlamaForCausalLM,
    batch: BatchInfo = None,
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
    logits = self.lm_head(hidden_states)
    return logits


def llama_model_forward(
    self: LlamaModel,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    """This function will replace the forward function of LlamaModel.

    Args:
        batch (BatchInfo, optional): It stores the necessary input information for this inference.. Defaults to None.
        k_caches (List[torch.Tensor], optional): It holds the GPU memory for the key cache. Defaults to None.
        v_caches (List[torch.Tensor], optional): It holds the GPU memory for the value cache. Defaults to None.
    """
    input_ids = batch.get_batch_inputs()
    block_tables = batch.get_block_table_tensor()
    attention_mask = batch.get_attn_mask()

    if attention_mask is not None:
        if HAS_TRITON:
            sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        else:
            sequence_lengths = batch.get_sequence_lengths()
    else:
        sequence_lengths = batch.get_sequence_lengths()

    batch_size, _ = input_ids.shape
    kv_seq_len = sequence_lengths.max().item()

    if attention_mask is not None:
        if batch.is_prompts:
            # Here, we generate position_ids through the input tensor, which can align with the output precision of the transformer.
            position_ids = generate_padding_position_id(attention_mask)
        else:
            position_ids = (attention_mask.sum(dim=-1) - 1).reshape(-1, 1)
    else:
        if batch.is_prompts:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=batch.device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.arange(kv_seq_len - 1, kv_seq_len, dtype=torch.long, device=batch.device)
            position_ids = position_ids.unsqueeze(0)

    hidden_states = self.embed_tokens(input_ids)

    cos_sin = get_xine_cache(sequence_lengths, self._cos_cached, self._sin_cached, batch.is_prompts)

    if batch.is_prompts:
        output_tensor = torch.zeros(
            (sequence_lengths.sum().item(), batch.num_heads, batch.head_dim), dtype=batch.dtype, device=batch.device
        )
    else:
        output_tensor = torch.zeros(
            (batch_size, batch.num_heads, batch.head_dim), dtype=batch.dtype, device=batch.device
        )
    sm_scale = 1.0 / (batch.head_dim**0.5)

    norm_output = torch.empty_like(hidden_states)

    for layer_id, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            position_ids=position_ids,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=batch.is_prompts,
            sequence_lengths=sequence_lengths,
            attention_mask=attention_mask,
            kv_seq_len=kv_seq_len,
            cos_sin=cos_sin,
            fd_inter_tensor=batch.fd_inter_tensor,
            output_tensor=output_tensor,
            norm_output=norm_output,
            sm_scale=sm_scale,
        )

    if batch.is_prompts:
        hidden_states = hidden_states[:, -1, :].unsqueeze(dim=1).contiguous()
        norm_output = torch.empty_like(hidden_states)
    hidden_states = self.norm(hidden_states.reshape(-1, hidden_states.shape[-1]), norm_output)

    return hidden_states


def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
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
        position_ids (torch.LongTensor), The position ids of input sequences.
        block_tables (torch.Tensor, optional): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
            storing mapping of token_position_id -> block_id. Defaults to None.
        k_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
        v_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
        is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
        sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence. Defaults to None.
        kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
        cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin. Defaults to None.
        fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for storing intermediate values in flash-decoding. Defaults to None.
        output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
        norm_output (torch.Tensor, optional): The mid tensor holds the output of layernorm. Defaults to None.
        sm_scale (int, optional): Used for flash attention. Defaults to None.
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states.reshape(-1, hidden_states.shape[-1]), norm_output)
    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        position_ids=position_ids,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        sequence_lengths=sequence_lengths,
        attention_mask=attention_mask,
        kv_seq_len=kv_seq_len,
        cos_sin=cos_sin,
        fd_inter_tensor=fd_inter_tensor,
        output_tensor=output_tensor,
        sm_scale=sm_scale,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states.reshape(-1, hidden_states.shape[-1]), norm_output)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


class PadLlamaAttention(LlamaAttention):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        attn_qproj_w: torch.nn.Parameter = None,
        attn_kproj_w: torch.nn.Parameter = None,
        attn_vproj_w: torch.nn.Parameter = None,
        attn_oproj_w: torch.nn.Parameter = None,
    ):
        """This layer will replace the LlamaAttention.

        Args:
            config (LlamaConfig): Holding the Llama model config.
            layer_idx (Optional[int], optional): The decode layer id of this attention layer. Defaults to None.
            attn_qproj_w (torch.nn.Parameter, optional): The q_proj weight. Defaults to None.
            attn_kproj_w (torch.nn.Parameter, optional): The k_proj weight. Defaults to None.
            attn_vproj_w (torch.nn.Parameter, optional): The v_proj weight. Defaults to None.
            attn_oproj_w (torch.nn.Parameter, optional): The o_proj weight. Defaults to None.
        """
        super().__init__(config, layer_idx)
        self.q_proj.weight = attn_qproj_w
        self.k_proj.weight = attn_kproj_w
        self.v_proj.weight = attn_vproj_w
        self.o_proj.weight = attn_oproj_w

    @staticmethod
    def from_native_module(module: LlamaAttention, *args, **kwargs) -> LlamaAttention:
        """Used for initialize the weight of NopadLlamaAttention by origin LlamaAttention

        Args:
            module (LlamaAttention): The origin LlamaAttention layer.
        """
        config = module.config
        layer_idx = module.layer_idx

        attn_qproj_w = module.q_proj.weight
        attn_kproj_w = module.k_proj.weight
        attn_vproj_w = module.v_proj.weight
        attn_oproj_w = module.o_proj.weight

        attn_layer = PadLlamaAttention(
            config=config,
            layer_idx=layer_idx,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj_w=attn_oproj_w,
        )

        return attn_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        block_tables: torch.Tensor = None,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        is_prompts: bool = True,
        sequence_lengths: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        kv_seq_len: int = 0,
        cos_sin: Tuple[torch.Tensor] = None,
        fd_inter_tensor: FDIntermTensors = None,
        output_tensor: torch.Tensor = None,
        sm_scale: int = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim]
            position_ids (torch.LongTensor), The position ids of input sequences.
            block_tables (torch.Tensor, optional): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
                storing mapping of token_position_id -> block_id. Defaults to None.
            k_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
            v_cache (torch.Tensor, optional): It holds the GPU memory for the key cache. Defaults to None.
            is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
            sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence. Defaults to None.
            attention_mask (torch.Tensor, optional): The padding mask - corresponds to a tensor of size [batch_size, seq_len]
                where 0 stands for the position of padding tokens and 1 for the position of non-padding tokens.
            kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
            cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin. Defaults to None.
            fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
                storing intermediate values in flash-decoding. Defaults to None.
            output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
            sm_scale (int, optional): Used for flash attention. Defaults to None.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if HAS_TRITON:
            if is_prompts:
                if attention_mask is not None:
                    query_states, key_states, value_states, indices = unpading_input(
                        query_states, key_states, value_states, attention_mask
                    )
                else:
                    query_states = query_states.view(-1, self.num_heads, self.head_dim)
                    key_states = key_states.view(-1, self.num_heads, self.head_dim)
                    value_states = value_states.view(-1, self.num_heads, self.head_dim)
            else:
                query_states = query_states.squeeze(dim=1)
                key_states = key_states.squeeze(dim=1)
                value_states = value_states.squeeze(dim=1)

            rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])

            block_size = k_cache.size(-2)

            if is_prompts:
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
                if attention_mask is not None:
                    attn_output = pad_input(attn_output, indices, bsz, q_len)
            else:
                copy_kv_to_blocked_cache(key_states, k_cache, kv_lengths=sequence_lengths, block_tables=block_tables)
                copy_kv_to_blocked_cache(value_states, v_cache, kv_lengths=sequence_lengths, block_tables=block_tables)
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
                )
                attn_output = attn_output.squeeze(1)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if is_prompts:
                attn_output = PagedAttention.pad_context_forward(
                    query_states,
                    key_states,
                    value_states,
                    k_cache,
                    v_cache,
                    sequence_lengths,
                    block_tables,
                    attention_mask,
                )
            else:
                attn_output = PagedAttention.pad_decoding_forward(
                    query_states,
                    key_states,
                    value_states,
                    k_cache,
                    v_cache,
                    sequence_lengths,
                    block_tables,
                    attention_mask,
                )

        attn_output = attn_output.view(bsz, q_len, self.num_heads, self.head_dim)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


def generate_padding_position_id(attention_mask: torch.Tensor) -> torch.Tensor:
    """Generate padding position_id through attention mask.

    Args:
        attention_mask (`torch.Tensor` of shape [batch_size, sequence_length]:
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

    Returns:
        torch.Tensor: The padding position_id.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def unpading_input(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor):
    """Convert padding input to nopad input.

    Args:
        q (torch.Tensor): [batch_size, q_seq_len, head_num, head_dim]
        k (torch.Tensor): [batch_size, q_seq_len, head_num, head_dim]
        v (torch.Tensor): [batch_size, q_seq_len, head_num, head_dim]
        attention_mask (torch.Tensor): [batch_size, sequence_length]

    Returns:
        Tuple[torch.Tensor]: The unpad q, k, v and The index of valid data in each batch.

    """
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    batch_size, kv_seq_len, num_key_value_heads, head_dim = q.shape
    q = index_first_axis(q.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    k = index_first_axis(k.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    v = index_first_axis(v.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices)
    return (q, k, v, indices)

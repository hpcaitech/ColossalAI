import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from colossalai.inference.modeling.layers.attention import PagedAttention, convert_kvcache, copy_to_cache


@pytest.mark.skip(reason="This test is not used in the current version.")
def test_copy_to_cache():
    key = torch.ones((2, 11, 3, 3))
    key[0, 9, :, :] = 0
    key[1, -2:, :, :] = 0
    cache = torch.zeros(8, 3, 8, 3)
    block_tables = torch.tensor([[0, 1], [2, 3]])
    lengths = torch.tensor([9, 8])
    cache = copy_to_cache(key, cache=cache, lengths=lengths, block_tables=block_tables, type="prefill")
    assert cache[1, 0, 0, 0] == 1
    assert cache[3, 0, 0, 0] == 0

    decoding_key = torch.ones((2, 1, 3, 3))
    cache = copy_to_cache(decoding_key, cache=cache, lengths=lengths + 1, block_tables=block_tables, type="decoding")
    assert cache[1, 0, 0, 1] == 1
    assert cache[3, 0, 0, 0] == 1


@pytest.mark.skip(reason="This test is not used in the current version.")
def test_convert_kvcache():
    cache = torch.ones(8, 3, 8, 3)
    key = torch.ones(2, 1, 3, 3) + 1
    lengths = torch.tensor([10, 9])
    block_tables = torch.tensor([[0, 1], [2, 3]])
    copy_to_cache(key, cache=cache, lengths=lengths, block_tables=block_tables, type="decoding")
    converted_cache = convert_kvcache(cache=cache, lengths=lengths, block_tables=block_tables)
    assert converted_cache.shape == (2, 10, 3, 3)


@pytest.mark.skip(reason="This test is not used in the current version.")
def test_context_attention():
    """
    test config: head_num = 4, head_size = 4
    """
    attn = PagedAttention()
    q = k = v = torch.randn(8, 4, 4)
    k_cache = torch.empty(8, 4, 8, 4)
    v_cache = torch.empty(8, 4, 8, 4)
    context_lengths = torch.tensor(
        [
            8,
        ]
    )
    block_tables = torch.tensor([[0, 1]])
    attn.nopad_context_forward(q, k, v, k_cache, v_cache, context_lengths, block_tables)
    # test padded q/k/v
    pad_q = pad_k = pad_v = q.unsqueeze(0)
    attn.pad_context_forward(pad_q, pad_k, pad_v, k_cache, v_cache, context_lengths, block_tables)

    config = LlamaConfig(num_attention_heads=4, num_key_value_heads=None, hidden_size=16)
    transformer_attn = LlamaAttention(config)
    transformer_attn.training = False

    # test accuracy with LlamaAttention
    hidden_states = torch.randn(1, 8, 16)
    proj_q = transformer_attn.q_proj(hidden_states).view(1, 8, 4, 4).transpose(1, 2)
    proj_k = transformer_attn.k_proj(hidden_states).view(1, 8, 4, 4).transpose(1, 2)
    proj_v = transformer_attn.v_proj(hidden_states).view(1, 8, 4, 4).transpose(1, 2)

    position_ids = torch.arange(0, 8, dtype=torch.long, device=proj_q.device)
    position_ids = position_ids.unsqueeze(0)
    cos, sin = transformer_attn.rotary_emb(proj_v, 8)
    proj_q, proj_k = apply_rotary_pos_emb(proj_q, proj_k, cos, sin, position_ids)

    pad_attn_output = attn.pad_context_forward(
        proj_q.transpose(1, 2),
        proj_k.transpose(1, 2),
        proj_v.transpose(1, 2),
        k_cache,
        v_cache,
        context_lengths,
        block_tables,
    )
    pad_attn_output = transformer_attn.o_proj(pad_attn_output)
    attn_mask = AttentionMaskConverter._make_causal_mask(
        hidden_states.shape[:2], q.dtype, q.device, past_key_values_length=0
    )
    attn_mask += PagedAttention.generate_padding_mask(context_lengths, 8)
    attn_output, _, _ = transformer_attn.forward(hidden_states, attention_mask=attn_mask)
    assert torch.allclose(pad_attn_output, attn_output, atol=1e-3, rtol=1e-3)


@pytest.mark.skip(reason="This test is not used in the current version.")
def test_decoding_attention():
    # test the pipeline of decoding attention
    attn = PagedAttention()
    q = k = v = torch.randn(2, 1, 4, 8)
    k_cache = torch.empty(8, 4, 8, 8)
    v_cache = torch.empty(8, 4, 8, 8)
    past_kv = torch.randn(2, 8, 4, 8)
    context_lenghths = torch.tensor([8, 8])
    lengths = context_lenghths + 1
    block_tables = torch.tensor([[0, 1], [2, 3]])
    copy_to_cache(past_kv, k_cache, lengths=context_lenghths, block_tables=block_tables)
    copy_to_cache(past_kv, v_cache, lengths=context_lenghths, block_tables=block_tables)
    attn.pad_decoding_forward(q, k, v, k_cache, v_cache, lengths=lengths, block_tables=block_tables)

    # test decoding accuracy, past_kv is reused
    config = LlamaConfig(num_attention_heads=4, num_key_value_heads=None, hidden_size=32)
    transformer_attn = LlamaAttention(config)
    transformer_attn.layer_idx = 0
    transformer_attn.training = False
    hidden_states = torch.randn(2, 1, 32)
    proj_q = transformer_attn.q_proj(hidden_states).view(2, 1, 4, 8).transpose(1, 2)
    proj_k = transformer_attn.k_proj(hidden_states).view(2, 1, 4, 8).transpose(1, 2)
    proj_v = transformer_attn.v_proj(hidden_states).view(2, 1, 4, 8).transpose(1, 2)

    cos, sin = transformer_attn.rotary_emb(proj_v, 16)
    position_ids = lengths - 1
    position_ids = position_ids.unsqueeze(1)  # NOTE: this may be wrong
    proj_q, proj_k = apply_rotary_pos_emb(proj_q, proj_k, cos, sin, position_ids, unsqueeze_dim=2)

    llama_past_kv = DynamicCache()
    llama_past_kv.update(key_states=past_kv.transpose(1, 2), value_states=past_kv.transpose(1, 2), layer_idx=0)

    # past_key_value shape in Llama: bsz, num_heads, seq_len, head_dim
    pad_attn_output = attn.pad_decoding_forward(
        proj_q.transpose(1, 2), proj_k.transpose(1, 2), proj_v.transpose(1, 2), k_cache, v_cache, lengths, block_tables
    )
    attn_mask = AttentionMaskConverter._make_causal_mask(q.shape[:2], q.dtype, q.device, past_key_values_length=8)
    attn_mask = attn_mask + PagedAttention.generate_padding_mask(lengths, 9).unsqueeze(1).unsqueeze(2)

    pad_attn_output = transformer_attn.o_proj(pad_attn_output)
    position_ids = context_lenghths.unsqueeze(1)
    attn_output, _, _ = transformer_attn.forward(
        hidden_states, past_key_value=llama_past_kv, position_ids=position_ids, attention_mask=attn_mask
    )
    assert torch.allclose(pad_attn_output, attn_output, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    test_copy_to_cache()
    test_convert_kvcache()
    test_context_attention()
    test_decoding_attention()

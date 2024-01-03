import pytest
import torch

import colossalai
from colossalai.inference.modeling.layers.attention import PagedAttention, convert_kvcache, copy_to_cache
from colossalai.testing import spawn


def test_copy_to_cache():
    key = torch.ones((2, 10, 3, 3))
    key[0, 9, :, :] = 0
    key[1, -2:, :, :] = 0
    cache = torch.zeros(8, 3, 3, 8)
    block_tables = torch.tensor([[0, 1], [2, 3]])
    lengths = torch.tensor([9, 8])
    cache = copy_to_cache(key, cache=cache, lengths=lengths, block_tables=block_tables, type="prefill")
    assert cache[1, 0, 0, 0] == 1
    assert cache[3, 0, 0, 0] == 0

    decoding_key = torch.ones((2, 1, 3, 3))
    cache = copy_to_cache(decoding_key, cache=cache, lengths=lengths + 1, block_tables=block_tables, type="decoding")
    assert cache[1, 0, 0, 1] == 1
    assert cache[3, 0, 0, 0] == 1


def test_convert_kvcache():
    cache = torch.ones(8, 3, 3, 8)
    key = torch.ones(2, 1, 3, 3) + 1
    lengths = torch.tensor([10, 9])
    block_tables = torch.tensor([[0, 1], [2, 3]])
    converted_cache = convert_kvcache(key, cache=cache, lengths=lengths, block_tables=block_tables)
    assert converted_cache.shape == (2, 10, 3, 3)


def test_context_attention():
    attn = PagedAttention(4, 4)
    q = k = v = torch.randn(8, 4, 4)
    k_cache = torch.empty(8, 4, 4, 8)
    v_cache = torch.empty(8, 4, 4, 8)
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


def test_decoding_attention():
    attn = PagedAttention(4, 4)
    q = k = v = torch.randn(2, 1, 4, 4)
    k_cache = torch.empty(8, 4, 4, 8)
    v_cache = torch.empty(8, 4, 4, 8)
    past_kv = torch.randn(2, 8, 4, 4)
    context_lenghths = torch.tensor([8, 8])
    block_tables = torch.tensor([[0, 1], [2, 3]])
    copy_to_cache(past_kv, k_cache, lengths=context_lenghths, block_tables=block_tables)
    copy_to_cache(past_kv, v_cache, lengths=context_lenghths, block_tables=block_tables)
    attn.pad_decoding_forward(q, k, v, k_cache, v_cache, lengths=context_lenghths + 1, block_tables=block_tables)


def check_attention_layer():
    test_copy_to_cache()
    test_convert_kvcache()
    test_context_attention()
    test_decoding_attention()


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_attention_layer()


@pytest.mark.dist
def test_attention_layer():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_attention_layer()

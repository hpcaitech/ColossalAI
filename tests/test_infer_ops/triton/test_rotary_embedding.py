import pytest
import torch
from transformers.models.llama import LlamaConfig

from colossalai.inference.config import InferenceConfig
from colossalai.inference.kv_cache import KVCacheManager
from colossalai.kernel.triton import rotary_embedding_fwd


def torch_rotary_emb(x, cos, sin):
    seq_len, h, dim = x.shape
    x0 = x[:, :, 0 : dim // 2]
    x1 = x[:, :, dim // 2 : dim]
    cos = cos.view((seq_len, 1, dim // 2))
    sin = sin.view((seq_len, 1, dim // 2))
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return torch.cat((o0, o1), dim=-1)


@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("SEQ_LEN", [64])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rotary_emb(BATCH_SIZE, SEQ_LEN, H, D, dtype, eps=1e-5, device="cuda"):
    TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
    kv_cache_config = {
        "hidden_size": H * D,
        "num_attention_heads": H,
        "num_layers": 1,
        "block_size": 4,
        "max_batch_size": 4,
        "max_input_len": 64,
        "max_output_len": 32,
        "dtype": torch.float16,
        "beam_width": 3,
        "tp_size": 1,
    }

    hidden_size = kv_cache_config.pop("hidden_size")
    num_layers = kv_cache_config.pop("num_layers")
    num_attention_heads = kv_cache_config.pop("num_attention_heads")
    hidden_size // num_attention_heads
    block_size = kv_cache_config["block_size"]
    max_batch_size = kv_cache_config["max_batch_size"]
    kv_cache_config["max_input_len"]
    kv_cache_config["max_output_len"]

    inference_config = InferenceConfig(model="", **kv_cache_config)
    model_config = LlamaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
    )
    cache_manager = KVCacheManager(inference_config, model_config)

    max_blocks_per_seq = cache_manager.get_max_blocks_per_sequence()
    block_tables = torch.tensor(
        [[-1 for _ in range(max_blocks_per_seq)] for _ in range(kv_cache_config["max_batch_size"])], dtype=torch.int32
    )

    context_lengths = [SEQ_LEN for _ in range(max_batch_size)]

    for req_i in range(max_batch_size):
        cur_seq_len = context_lengths[req_i]
        cur_block_table = block_tables[req_i]
        cache_manager.allocate_context_from_block_table(cur_block_table, cur_seq_len)

    allocated_blocks = SEQ_LEN // block_size
    last_block_tokens = SEQ_LEN % block_size

    tokens_ids = []
    for i in range(BATCH_SIZE):
        for j in range(allocated_blocks):
            tokens_ids.extend([block_tables[i][j].item() * block_size + k for k in range(block_size)])

        for k in range(last_block_tokens):
            tokens_ids.append(block_tables[i][allocated_blocks + 1].item() * block_size + k)

    tokens_offset = torch.tensor(tokens_ids, dtype=torch.int32, device="cuda")

    # create data
    q_shape = (TOTAL_TOKENS, H, D)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (TOTAL_TOKENS, H, D)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    cos_shape = (TOTAL_TOKENS, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")

    q_ref = torch_rotary_emb(q, cos, sin)
    k_ref = torch_rotary_emb(k, cos, sin)

    key_cache = cache_manager._kv_caches[0][0]
    rotary_embedding_fwd(q, k, cos, sin, key_cache, tokens_offset)
    q_tri = q
    k_tri = torch.index_select(key_cache.view(-1, H, D), 0, tokens_offset)

    assert torch.allclose(q_tri, q_ref, atol=1e-2, rtol=1e-3)
    assert torch.allclose(k_tri, k_ref, atol=1e-2, rtol=1e-3)


if __name__ == "__main__":
    test_rotary_emb(1024, 32, 64, torch.float16)

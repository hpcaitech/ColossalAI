import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

from colossalai.kernel.kernel_loader import InferenceOpsLoader
inference_ops = InferenceOpsLoader().load()

from tests.test_infer.test_ops.triton.test_rotary_embdding_unpad import torch_rotary_emb

@pytest.mark.parametrize("BATCH_SIZE", [4])
@pytest.mark.parametrize("SEQ_LEN", [64])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("D", [64])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rotary_emb(BATCH_SIZE, SEQ_LEN, H, D, dtype):
    torch.manual_seed(10)
    TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
    # our crafted op equals to Transformers
    x0 = torch.randn(TOTAL_TOKENS, SEQ_LEN, D, dtype=dtype)
    x1 = torch.randn(TOTAL_TOKENS, SEQ_LEN, D, dtype=dtype)
    emb = LlamaRotaryEmbedding(D)
    cos, sin = emb(x0, TOTAL_TOKENS)
    cos_2 = cos[:, : D // 2]
    sin_2 = sin[:, : D // 2]
    position_ids = torch.arange(TOTAL_TOKENS)
    embd_x0, _ = apply_rotary_pos_emb(x0, x1, cos, sin, position_ids)
    embd_stimulated_x = torch_rotary_emb(x0, cos_2, sin_2)
    assert torch.allclose(embd_x0, embd_stimulated_x)

    # create data
    cos_shape = (TOTAL_TOKENS, D // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    new_k = torch.randn((BATCH_SIZE, H, D), dtype=dtype, device="cuda")
    new_q = torch.randn_like(new_k)

    q_ref = torch_rotary_emb(new_q, cos[:BATCH_SIZE], sin[:BATCH_SIZE])
    k_ref = torch_rotary_emb(new_k, cos[:BATCH_SIZE], sin[:BATCH_SIZE])

    inference_ops.rotary_embedding(new_q, new_k, cos, sin)
    
    assert torch.allclose(new_q, q_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(new_k, k_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_rotary_emb(16, 128, 128, 512, torch.float16)

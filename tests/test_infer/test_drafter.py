import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from colossalai.inference.modeling.models.glide_llama import GlideLlamaConfig, GlideLlamaForCausalLM
from colossalai.inference.spec.drafter import Drafter
from colossalai.utils import get_current_device

NUM_LAYERS = 1
MAX_LEN = 100
SPEC_NUM = 5


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


@pytest.mark.parametrize("spec_num", [SPEC_NUM])
def test_drafter(tokenizer, spec_num: int):
    torch.manual_seed(123)

    device = get_current_device()
    toy_config = LlamaConfig(num_hidden_layers=NUM_LAYERS)
    toy_config.pad_token_id = tokenizer.eos_token_id
    drafter_model = LlamaForCausalLM(toy_config)
    drafter_model = drafter_model.eval().cuda()

    drafter = Drafter(drafter_model, tokenizer, device=device)

    input_ids = torch.randint(low=5, high=1000, size=(1, 6)).to(device)
    out = drafter.speculate(input_ids, spec_num)
    past_kv_length = input_ids.size(1) + spec_num - 1

    assert out.speculated_length == spec_num
    assert out.next_tokens.shape == (spec_num,)
    assert out.logits.shape == (spec_num, len(tokenizer))
    assert out.past_key_values[0][0].size(2) == past_kv_length

    reject_num = max(0, spec_num - 1)
    trimmed_past_key_values = drafter.trim_kv_cache(out.past_key_values, reject_num)
    assert trimmed_past_key_values[0][0].size(2) == past_kv_length - reject_num


def test_spec_dec(tokenizer):
    spec_num = SPEC_NUM
    device = get_current_device()
    tokenizer.pad_token = tokenizer.eos_token

    # Dummy config for Glide Model
    glide_config = GlideLlamaConfig(
        intermediate_size=8192,
        large_hidden_size=4096,
        large_num_attention_heads=32,
        num_hidden_layers=NUM_LAYERS,
    )
    drafter_model = GlideLlamaForCausalLM(glide_config)

    assert hasattr(drafter_model, "model")
    assert hasattr(drafter_model.model, "layers")
    for _, layer in enumerate(drafter_model.model.layers):
        assert hasattr(layer, "cross_attn")

    # Init the Drafter by providing the sharded drafter model
    drafter = Drafter(drafter_model, tokenizer, device=device, dtype=torch.float16)

    input_ids = torch.randint(low=5, high=1000, size=(1, 6)).to(device)
    out = drafter.speculate(input_ids, spec_num, past_key_values=None)


if __name__ == "__main__":
    dummy_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    test_drafter(dummy_tokenizer, spec_num=SPEC_NUM)
    test_spec_dec(dummy_tokenizer)

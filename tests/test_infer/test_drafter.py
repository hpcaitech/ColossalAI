import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from colossalai.inference.spec.drafter import Drafter
from colossalai.utils import get_current_device

NUM_LAYERS = 2


@pytest.mark.parametrize("spec_num", [5])
def test_drafter(spec_num: int):
    torch.manual_seed(123)

    device = get_current_device()

    toy_config = LlamaConfig(num_hidden_layers=NUM_LAYERS)
    toy_config.pad_token_id = toy_config.eos_token_id
    drafter_model = LlamaForCausalLM(toy_config)
    drafter_model = drafter_model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    drafter = Drafter(drafter_model, tokenizer, spec_num, device=device)

    input_ids = torch.randint(low=5, high=1000, size=(1, 6)).to(device)
    out = drafter.speculate(input_ids, spec_num)
    past_kv_length = input_ids.size(1) + spec_num - 1

    assert out.speculated_length == spec_num
    assert out.next_tokens.shape == (spec_num,)
    assert out.logits.shape == (spec_num, len(tokenizer))
    assert drafter._past_key_values[0][0].size(2) == out.past_key_values[0][0].size(2) == past_kv_length

    reject_num = 3
    assert reject_num <= spec_num
    drafter.trim_kv_cache(reject_num)
    assert drafter._past_key_values[0][0].size(2) == past_kv_length - reject_num


if __name__ == "__main__":
    test_drafter(spec_num=5)

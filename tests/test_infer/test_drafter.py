import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.config import GenerationConfig, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.inference.spec.drafter import Drafter
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device

NUM_LAYERS = 2
MAX_LEN = 100


@pytest.mark.parametrize("spec_num", [5])
def test_drafter(spec_num: int):
    torch.manual_seed(123)

    device = get_current_device()

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
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


def check_sd():
    torch.manual_seed(123)

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    # Dummy configs for testing
    toy_config = LlamaConfig(num_hidden_layers=NUM_LAYERS)
    toy_config.pad_token_id = tokenizer.eos_token_id
    drafter_model = LlamaForCausalLM(toy_config)
    drafter_model = drafter_model.eval().cuda()
    large_config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_attention_heads=32,
        num_hidden_layers=8,
        num_key_value_heads=32,
        max_position_embeddings=2048,
    )
    large_config.pad_token_id = tokenizer.eos_token_id
    main_model = LlamaForCausalLM(large_config)

    inference_config = InferenceConfig(
        dtype="fp16",
        micro_batch_size=1,
        max_batch_size=1,
        max_input_len=128,
        max_output_len=128,
        prefill_ratio=1.2,
        block_size=16,
    )
    engine = InferenceEngine(main_model, tokenizer, inference_config)
    engine.enable_spec_dec(drafter_model, n_spec_tokens=5)

    dummy_inputs = torch.randint(low=5, high=1000, size=(1, 10), dtype=torch.long, device="cuda")
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        max_length=MAX_LEN,
        eos_token_id=tokenizer.eos_token_id,
    )
    out, out_token_ids = engine.generate(
        prompts_token_ids=dummy_inputs, generation_config=generation_config, return_token_ids=True
    )
    engine.disable_spec_dec()
    engine.clear_spec_dec()

    assert not engine.use_spec_dec
    assert engine.drafter is None and engine.drafter_model is None

    assert len(out) == 1
    assert len(out_token_ids) == 1 and len(out_token_ids[0]) == MAX_LEN


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_sd()


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_spec_dec():
    spawn(run_dist, nprocs=1)


if __name__ == "__main__":
    test_drafter(spec_num=5)
    test_spec_dec()

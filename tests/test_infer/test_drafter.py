import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import colossalai
from colossalai.inference.modeling.models.glide_llama import GlideLlamaConfig
from colossalai.inference.modeling.policy import model_policy_map
from colossalai.inference.spec.drafter import Drafter
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device

NUM_LAYERS = 2
MAX_LEN = 100
SPEC_NUM = 5


@pytest.mark.parametrize("spec_num", [SPEC_NUM])
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


def check_shard_drafter():
    spec_num = SPEC_NUM
    device = get_current_device()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    # Test Glide Llama Modeling and Sharding
    model_policy_name = "glide_llama"

    # Dummy config for Glide Model
    glide_config = GlideLlamaConfig(
        intermediate_size=8192,
        large_hidden_size=4096,
        large_num_attention_heads=32,
        num_hidden_layers=NUM_LAYERS,
    )
    drafter_model = LlamaForCausalLM(glide_config)

    # Use shardformer to replace layers of the drafter model
    shard_config = ShardConfig(
        tensor_parallel_process_group=None,
        pipeline_stage_manager=None,
        enable_tensor_parallelism=False,
        enable_fused_normalization=False,
        enable_all_optimization=False,
        enable_flash_attention=False,
        enable_jit_fused=False,
        enable_sequence_parallelism=False,
    )
    shardformer = ShardFormer(shard_config=shard_config)
    model_policy = model_policy_map[model_policy_name]
    drafter_model, _ = shardformer.optimize(drafter_model, model_policy())

    assert hasattr(drafter_model, "model")
    assert hasattr(drafter_model.model, "layers")
    for _, layer in enumerate(drafter_model.model.layers):
        assert hasattr(layer, "cross_attn")

    # Init the Drafter by providing the sharded drafter model
    drafter = Drafter(drafter_model, tokenizer, device=device, dtype=torch.float16)

    input_ids = torch.randint(low=5, high=1000, size=(1, 6)).to(device)
    out = drafter.speculate(input_ids, spec_num, past_key_values=None)


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    check_spec_dec()
    check_shard_drafter()


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_spec_dec():
    spawn(run_dist, nprocs=1)


if __name__ == "__main__":
    test_drafter(spec_num=SPEC_NUM)
    test_spec_dec()

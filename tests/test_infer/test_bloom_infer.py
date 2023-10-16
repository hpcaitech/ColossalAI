import pytest
import torch
from packaging import version
from transformers import BloomForCausalLM
from transformers.models.bloom.configuration_bloom import BloomConfig

import colossalai
from colossalai.inference.tensor_parallel import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

TP_SIZE = 2
MAX_BATCH_SIZE = 4
MAX_INPUT_LEN = 16
MAX_OUTPUT_LEN = 32

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


@parameterize(
    "test_config",
    [
        {
            "tp_size": TP_SIZE,
        }
    ],
)
def run(test_config):
    bloom_config = BloomConfig(num_hidden_layers=2, bos_token_id=0, eos_token_id=1, vocab_size=1200, hidden_size=1024)
    model = BloomForCausalLM(bloom_config)
    model = model.half()

    shard_config = ShardConfig(
        enable_tensor_parallelism=True if test_config["tp_size"] > 1 else False, inference_only=True
    )
    infer_engine = TPInferEngine(model, shard_config, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)

    input_tokens = {
        "input_ids": torch.randint(1, 1000, (MAX_BATCH_SIZE, MAX_INPUT_LEN), device="cuda"),
        "attention_mask": torch.ones((MAX_BATCH_SIZE, MAX_INPUT_LEN), device="cuda"),
    }
    outputs = infer_engine.generate(input_tokens, **generate_kwargs)

    assert outputs is not None


def check_bloom(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom_infer():
    spawn(check_bloom, TP_SIZE)


if __name__ == "__main__":
    test_bloom_infer()

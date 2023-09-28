import os

import pytest
import torch
from packaging import version
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
TPSIZE = 2
BATCH_SIZE = 8
MAX_INPUT_LEN = 12
MAX_OUTPUT_LEN = 100

CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.5")


def init_to_get_rotary(self, base=10000):
    self.config.head_dim_ = self.config.hidden_size // self.config.num_attention_heads
    if not hasattr(self.config, "rope_scaling"):
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = self.config.rope_scaling.factor if self.config.rope_scaling is not None else 1.0
    if hasattr(self.config, "max_sequence_length"):
        max_seq_len = self.config.max_sequence_length
    elif hasattr(self.config, "max_position_embeddings"):
        max_seq_len = self.config.max_position_embeddings * rope_scaling_factor
    else:
        max_seq_len = 2048 * rope_scaling_factor
    base = float(base)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, self.config.head_dim_, 2, device="cpu", dtype=torch.float32) / self.config.head_dim_)
    )
    t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
    freqs = torch.outer(t, inv_freq)

    self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
    self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
    return


@parameterize(
    "test_config",
    [
        {
            "tp_size": TPSIZE,
        }
    ],
)
def run_llama_test(test_config):
    llama_config = LlamaConfig(num_hidden_layers=2, bos_token_id=0, eos_token_id=1, vocab_size=1200, hidden_size=1024)
    model = LlamaForCausalLM(llama_config)
    model = model.half()

    shard_config = ShardConfig(
        enable_tensor_parallelism=True if test_config["tp_size"] > 1 else False, inference_only=True
    )
    infer_engine = TPInferEngine(model, shard_config, BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
    init_to_get_rotary(model.model, base=10000)
    generate_kwargs = dict(max_new_tokens=MAX_OUTPUT_LEN, do_sample=False)

    input_tokens = {
        "input_ids": torch.randint(1, 1000, (BATCH_SIZE, MAX_INPUT_LEN), device="cuda"),
        "attention_mask": torch.ones((BATCH_SIZE, MAX_INPUT_LEN), device="cuda"),
    }
    outputs = infer_engine.generate(input_tokens, **generate_kwargs)

    assert outputs is not None


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_llama_test()


@pytest.mark.skipif(not CUDA_SUPPORT, reason="kv-cache manager engine requires cuda version to be higher than 11.5")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, TPSIZE)


if __name__ == "__main__":
    test_llama()

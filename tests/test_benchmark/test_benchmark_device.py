"""Self-tests of the benchmark framework on CUDA or NPU devices: allocator statistics, multi-rank runs and the
real-size model_zoo declarations."""

import pytest
import torch

from colossalai.accelerator import get_accelerator
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark.measure import Setup
from tests.kit.benchmark.run import run_benchmark
from tests.test_benchmark.test_benchmark_core import TINY


def _devices() -> int:
    accelerator = get_accelerator()
    return 0 if accelerator.name == "cpu" else accelerator.device_count()


def _setup(config, seq_len=16, **kwargs):
    return Setup(
        entry="tiny",
        size="tiny",
        config=config,
        seq_len=seq_len,
        micro_batch=2,
        warmup_steps=1,
        timed_steps=3,
        **kwargs,
    )


@pytest.mark.skipif(_devices() < 1, reason="needs a device")
@pytest.mark.parametrize("config", ["zero2_bf16", "gemini_bf16", "fsdp_bf16", "ddp_fp16"])
@rerun_if_address_is_in_use()
def test_memory_statistics_are_ordered(config):
    result = run_benchmark(_setup(config), nproc=1, repeats=1, spec=TINY)
    memory = result["memory_bytes"]
    assert 0 < memory["allocated_peak"] <= memory["reserved_peak"] <= memory["device_peak"] <= memory["device_total"]
    assert memory["setup_allocated"] <= memory["allocated_peak"]
    assert result["repeats"][0]["loss"]["finite"]
    assert result["environment"]["device_name"]


@pytest.mark.skipif(_devices() < 1, reason="needs a device")
@rerun_if_address_is_in_use()
def test_gradient_checkpointing_lowers_the_activation_peak():
    # a long sequence through eager attention, so that the activations kept for backward dominate the peak
    plain = run_benchmark(_setup("zero2_bf16", seq_len=2048), nproc=1, repeats=1, spec=TINY)
    checkpointed = run_benchmark(
        _setup("zero2_bf16", seq_len=2048, gradient_checkpointing=True), nproc=1, repeats=1, spec=TINY
    )
    assert checkpointed["setup"]["gradient_checkpointing"]
    assert checkpointed["memory_bytes"]["allocated_peak"] < 0.8 * plain["memory_bytes"]["allocated_peak"]


@pytest.mark.skipif(_devices() < 2, reason="needs two devices")
@rerun_if_address_is_in_use()
def test_two_ranks_count_every_rank_tokens():
    result = run_benchmark(_setup("zero2_bf16"), nproc=2, repeats=1, spec=TINY)
    assert result["world_size"] == 2
    assert result["repeats"][0]["tokens_per_step"] == 2 * 2 * 16
    assert result["throughput"]["tokens_per_s"] == pytest.approx(2 * result["throughput"]["tokens_per_s_per_device"])


# parameter counts of the published checkpoints, in billions
EXPECTED_PARAMETERS = {
    ("transformers_llama_for_causal_lm", "1b"): 1.24,
    ("transformers_llama_for_causal_lm", "8b"): 8.03,
    ("transformers_qwen3_for_causal_lm", "0.6b"): 0.60,
    ("transformers_qwen3_for_causal_lm", "1.7b"): 1.72,
    ("transformers_qwen3_for_causal_lm", "8b"): 8.19,
}


def test_real_sizes_match_the_published_models():
    # needs the whole model_zoo (timm, torchvision, ...), which only the GPU image installs
    from tests.kit.model_zoo import model_zoo

    declared = {
        (name, size): fn
        for name, entry in model_zoo.items()
        if entry[4] is not None and entry[4].benchmark is not None
        for size, fn in entry[4].benchmark.models.items()
    }
    assert set(EXPECTED_PARAMETERS) <= set(declared)
    for key, billions in EXPECTED_PARAMETERS.items():
        with torch.device("meta"):
            model = declared[key]()
        assert sum(p.numel() for p in model.parameters()) / 1e9 == pytest.approx(billions, abs=0.01), key

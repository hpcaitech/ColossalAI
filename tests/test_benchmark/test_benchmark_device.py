"""Self-tests of the benchmark framework on CUDA or NPU devices: allocator statistics, multi-rank runs and the
real-size model_zoo declarations."""

import pytest
import torch

from colossalai.accelerator import get_accelerator
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark import BenchmarkSpec
from tests.kit.benchmark.flops import counted_flops_per_token
from tests.kit.benchmark.measure import Setup
from tests.kit.benchmark.peak import lookup
from tests.kit.benchmark.run import run_benchmark, with_baseline
from tests.kit.benchmark.scaling import run_scaling
from tests.kit.benchmark.search import Probe
from tests.test_benchmark.test_benchmark_core import TINY, tiny_llama


def tiny_eager():
    return tiny_llama("eager")


# eager attention materializes the score matrix, so that long sequences need memory: for checkpointing and OOM tests
EAGER = BenchmarkSpec(models={"tiny": tiny_eager}, seq_len=16, micro_batch=2)


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
    if lookup(result["environment"]["device_name"], result["peak"]["dtype"]) is not None:
        assert 0 < result["throughput"]["mfu_nominal"] < result["throughput"]["mfu_measured"] < 1


@pytest.mark.skipif(_devices() < 1, reason="needs a device")
@rerun_if_address_is_in_use()
def test_gradient_checkpointing_lowers_the_activation_peak():
    # a long sequence through eager attention, so that the activations kept for backward dominate the peak
    plain = run_benchmark(_setup("zero2_bf16", seq_len=2048), nproc=1, repeats=1, spec=EAGER)
    checkpointed = run_benchmark(
        _setup("zero2_bf16", seq_len=2048, gradient_checkpointing=True), nproc=1, repeats=1, spec=EAGER
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


def _declared():
    # needs the whole model_zoo (timm, torchvision, ...), which only the GPU image installs
    from tests.kit.model_zoo import model_zoo

    return {
        (name, size): (entry[4].benchmark, fn)
        for name, entry in model_zoo.items()
        if entry[4] is not None and entry[4].benchmark is not None
        for size, fn in entry[4].benchmark.models.items()
    }


def test_real_sizes_match_the_published_models():
    declared = {key: fn for key, (_, fn) in _declared().items()}
    assert set(EXPECTED_PARAMETERS) <= set(declared)
    for key, billions in EXPECTED_PARAMETERS.items():
        with torch.device("meta"):
            model = declared[key]()
        assert sum(p.numel() for p in model.parameters()) / 1e9 == pytest.approx(billions, abs=0.01), key


def test_flops_formulas_match_the_counted_flops():
    for (name, size), (spec, fn) in _declared().items():
        if spec.flops_fn is None:
            continue
        with torch.device("meta"):
            formula = spec.flops_fn(fn(), spec.seq_len)
        counted = counted_flops_per_token(spec, size, spec.seq_len, micro_batch=1)
        assert counted == pytest.approx(formula, rel=0.02), (name, size)


@pytest.mark.skipif(_devices() < 1, reason="needs a device")
@rerun_if_address_is_in_use()
def test_a_real_out_of_memory_is_recognized():
    # eager attention over a million tokens: the score matrix alone is terabytes
    assert Probe(spec=EAGER)(_setup("zero2_bf16", seq_len=2**20)) == {"fits": False, "reason": "out of device memory"}
    assert Probe(spec=EAGER)(_setup("zero2_bf16"))["fits"]


@pytest.mark.skipif(_devices() < 2, reason="needs two devices")
@rerun_if_address_is_in_use()
def test_scaling_reads_communication_from_the_profiler():
    result = run_scaling(_setup("zero2_bf16"), counts=(1, 2), repeats=1, spec=TINY)
    one, two = result["scaling"]
    assert one["devices"] == 1 and two["devices"] == 2
    if get_accelerator().name == "cuda":
        # ZeRO-2 on two devices reduce-scatters gradients and all-gathers parameters every step
        assert 0 < two["exposed_communication"] <= two["communication"] < 1


@pytest.mark.skipif(_devices() < 2, reason="needs two devices")
@rerun_if_address_is_in_use()
def test_fsdp2_baseline_trains_and_compares():
    baseline = run_benchmark(_setup("pytorch_fsdp2_bf16"), nproc=2, repeats=1, spec=TINY)
    assert baseline["repeats"][0]["loss"]["finite"] and baseline["optimizer"] == "AdamW"
    memory = baseline["memory_bytes"]
    assert 0 < memory["allocated_peak"] <= memory["reserved_peak"]
    ours = run_benchmark(_setup("zero2_bf16"), nproc=2, repeats=1, spec=TINY)
    assert with_baseline(ours, baseline)["baseline"]["throughput_ratio"] > 0

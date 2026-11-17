"""Self-tests of the benchmark framework on CUDA or NPU devices: allocator statistics, multi-rank runs and the
real-size model_zoo declarations."""

import pytest
import torch
import transformers

from colossalai.accelerator import get_accelerator
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.benchmark import BenchmarkSpec
from tests.kit.benchmark.config.model import RANDOM, spec_of
from tests.kit.benchmark.config.setup import Setup
from tests.kit.benchmark.metrics.compute import counted_flops_per_token, lookup
from tests.kit.benchmark.scenarios.compare import run_compare
from tests.kit.benchmark.scenarios.scaling import run_scaling
from tests.kit.benchmark.scenarios.search import probe
from tests.kit.benchmark.scenarios.single import run_single
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
        weights=RANDOM,
        **kwargs,
    )


@pytest.mark.skipif(_devices() < 1, reason="needs a device")
@pytest.mark.parametrize(
    "config", ["zero2_bf16", "zero1_bf16_offload", "gemini_bf16", "fsdp_bf16", "ddp_fp16", "hybrid_zero2_bf16"]
)
@rerun_if_address_is_in_use()
def test_memory_statistics_are_ordered(config):
    measurement = run_single(_setup(config), repeats=1, spec=TINY)
    summary = measurement["summary"]
    memory = summary["memory_bytes"]
    assert 0 < memory["allocated_peak"] <= memory["reserved_peak"] <= memory["device_peak"] <= memory["device_total"]
    assert memory["setup_allocated"] <= memory["allocated_peak"]
    assert summary["repeats"][0]["loss_finite"]
    assert 0 < summary["time"]["forward_backward_share"] < 1
    device = measurement["runs"][0]["environment"]["device_name"]
    assert device
    if lookup(device, measurement["peak"]["dtype"]) is not None:
        assert 0 < summary["utilization"]["mfu_nominal"] < summary["utilization"]["mfu_measured"] < 1


@pytest.mark.skipif(_devices() < 1, reason="needs a device")
@rerun_if_address_is_in_use()
def test_gradient_checkpointing_lowers_the_activation_peak():
    # a long sequence through eager attention, so that the activations kept for backward dominate the peak
    plain = run_single(_setup("zero2_bf16", seq_len=2048), repeats=1, spec=EAGER)
    checkpointed = run_single(_setup("zero2_bf16", seq_len=2048, gradient_checkpointing=True), repeats=1, spec=EAGER)
    assert checkpointed["setup"]["gradient_checkpointing"]
    peak = "allocated_peak"
    assert checkpointed["summary"]["memory_bytes"][peak] < 0.8 * plain["summary"]["memory_bytes"][peak]


@pytest.mark.skipif(_devices() < 2, reason="needs two devices")
@rerun_if_address_is_in_use()
def test_two_ranks_count_every_rank_tokens():
    summary = run_single(_setup("zero2_bf16", devices=2), repeats=1, spec=TINY)["summary"]
    assert summary["devices"] == 2
    assert summary["time"]["tokens_per_step"] == 2 * 2 * 16
    assert summary["time"]["tokens_per_s"] == pytest.approx(2 * summary["time"]["tokens_per_s_per_device"])


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
    assert probe(_setup("zero2_bf16", seq_len=2**20), EAGER) == {"fits": False, "reason": "out of device memory"}
    assert probe(_setup("zero2_bf16"), EAGER)["fits"]


@pytest.mark.skipif(_devices() < 2, reason="needs two devices")
@rerun_if_address_is_in_use()
def test_scaling_reads_communication_from_the_profiler():
    one, two = run_scaling(_setup("zero2_bf16"), counts=(1, 2), repeats=1, spec=TINY)["curve"]
    assert one["devices"] == 1 and two["devices"] == 2
    if get_accelerator().name == "cuda":
        # ZeRO-2 on two devices reduce-scatters gradients and all-gathers parameters every step
        assert 0 < two["exposed_communication"] <= two["communication"] < 1


@pytest.mark.skipif(_devices() < 2, reason="needs two devices")
@rerun_if_address_is_in_use()
def test_fsdp2_baseline_trains_and_compares():
    ours = run_single(_setup("zero2_bf16", devices=2), repeats=1, spec=TINY)
    compared = run_compare(ours, "config", "pytorch_fsdp2_bf16", repeats=1, spec=TINY)
    baseline = compared["measurement"]["summary"]
    assert baseline["repeats"][0]["loss_finite"] and baseline["optimizer"] == "AdamW" and baseline["devices"] == 2
    memory = baseline["memory_bytes"]
    assert 0 < memory["allocated_peak"] <= memory["reserved_peak"]
    assert compared["ratios"]["throughput_ratio"] > 0


def test_benchmark_only_entries_and_the_moe_formula():
    from tests.kit.benchmark.entries import ENTRIES
    from tests.kit.benchmark.metrics.compute import moe_decoder_flops_per_token

    assert spec_of("qwen3_moe") is ENTRIES["qwen3_moe"] and spec_of("wan2_2_ti2v").searchable == ("micro_batch",)
    # Qwen3-30B-A3B, from its config.json: 30.5 B parameters, 3.3 B of them active per token
    config = transformers.Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=8,
        tie_word_embeddings=False,
    )
    with torch.device("meta"):
        model = transformers.Qwen3MoeForCausalLM(config)
    assert sum(p.numel() for p in model.parameters()) / 1e9 == pytest.approx(30.53, abs=0.01)
    # without attention, 6 FLOPs per active weight: the published 3.3 B active parameters less the input embedding
    # (0.31 B, a lookup) and the norms
    active = moe_decoder_flops_per_token(model, 0) / 6
    assert active / 1e9 == pytest.approx(3.3 - 0.31, abs=0.1)

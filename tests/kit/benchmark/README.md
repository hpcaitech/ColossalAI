# Benchmark

A shared framework for measuring how fast and how memory-hungry training a model is under a ColossalAI plugin, the
same way for every model, so that reports from different models, plugins and machines can be compared.

Covered so far: throughput and peak memory. Not covered yet, and listed as such in every report: MFU, the maximum
sequence length and batch size on one device, data-parallel scaling efficiency, and comparison with a baseline.

## Quick start

```bash
python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16 \
    --micro-batch 2 --out llama-1b.json
```

This trains Llama-3.2-1B (randomly initialized, real size) for 5 warmup and 20 timed steps, twice, each time in fresh
processes, and writes `llama-1b.json` and the rendered `llama-1b.md`. On H20 the summary line reads:

```
12,526 tokens/s on 1 × NVIDIA H20 (12,526 per device), peak reserved memory 48.30 GiB.
```

Options: `--nproc N` (one process per device), `--seq-len`, `--micro-batch` (per device), `--gradient-checkpointing`,
`--warmup-steps`, `--timed-steps`, `--repeats`. `--config` is one of:

| Config | Plugin |
| --- | --- |
| `ddp_fp32`, `ddp_fp16` | `TorchDDPPlugin`, without or with `mixed_precision="fp16"` |
| `zero1_bf16`, `zero2_bf16` | `LowLevelZeroPlugin(stage=1 or 2, precision="bf16")` |
| `zero2_bf16_offload` | the same with `cpu_offload=True` |
| `gemini_bf16` | `GeminiPlugin(precision="bf16")` |
| `fsdp_fp32`, `fsdp_bf16` | `TorchFSDPPlugin`, without or with bf16 `MixedPrecision` |

The names are those of the numerical alignment tool (`tests/kit/numerics/acceptance.py`), so a model's alignment and
benchmark reports refer to the same setups. The optimizer is `torch.optim.AdamW`, translated to `HybridAdam` for Gemini
(which accepts nothing else) as in the alignment tool; the report names the one used.

Render a JSON report again, e.g. after collecting it from another machine:

```bash
python -m tests.kit.benchmark.report llama-1b.json > llama-1b.md
```

Run on idle devices. Other processes slow the steps down and count towards the device-side memory; the report records
how much device memory was in use when the run started (this process's own context included, a few hundred MiB on
CUDA).

## Adding a model

Declare the real sizes on the model_zoo entry, next to its reduced test config:

```python
from tests.kit.benchmark import BenchmarkSpec

benchmark = BenchmarkSpec(
    models={
        "1b": lambda: transformers.LlamaForCausalLM(LlamaConfig(...)),  # from the published config.json
        "8b": lambda: transformers.LlamaForCausalLM(LlamaConfig(...)),
    },
    seq_len=4096,
)
model_zoo.register(..., model_attribute=ModelAttribute(benchmark=benchmark))
```

For a causal language model that is all. Other models also give `batch_fn` (one synthetic batch for `model(**batch)`),
`tokens_fn` (the tokens a batch counts for throughput, e.g. latent tokens of a diffusion model) and `loss_fn`; see
`spec.py`. Weights are random: throughput and memory of a dense model do not depend on them.

## What is measured, and how

Every model goes through the same protocol (`measure.py`):

- **Data**: each rank builds one batch of fixed shape, seeded by its rank, and feeds it to every step, so data loading
  is not measured.
- **Step**: forward, `booster.backward`, `optimizer.step`, `optimizer.zero_grad`. The device is synchronized right
  before and right after each step, so its time is the device's work and not only the host launching it.
- **Steps**: warmup steps are discarded. The time of a timed step is the maximum over ranks.
- **Repeats**: each repeat runs in fresh processes. Two repeats whose median step times differ by more than 5% are
  flagged in the report: the machine was not stable enough to compare the numbers with anything.

| Metric | Definition |
| --- | --- |
| Throughput | tokens of all ranks per step ÷ median step time; also per device and in samples/s. The median keeps one stalled step from moving the headline; p10 and p90 are reported next to it |
| Peak allocated | `max_memory_allocated`: memory held by tensors |
| Peak reserved | `max_memory_reserved`: memory held by the caching allocator, which is what runs out first |
| Peak on device | peak reserved plus what the process holds outside the allocator (CUDA context, communication buffers), measured at the end of the run: the device-side number `nvidia-smi` or `npu-smi` shows |
| After setup | allocated once the model and optimizer are boosted: weights, master weights and optimizer state before the first step |
| Host RSS peak | peak resident memory of a process, model construction included; what CPU offloading costs |

Peak memory is tracked from the first step on and is the maximum over ranks. All device calls go through
`colossalai.accelerator`, so the same code runs on CUDA and Ascend NPU; on CPU only the host memory is reported.

## Report

`run.py` writes one JSON file per setup (`schema_version` 1): the setup, every timed step of every repeat, the memory
peaks, the summary statistics, the environment (device, driver, clock and power limits, library versions), the
repository commit and whether the tree was dirty, and the command that reproduces it. The Markdown is rendered from the
JSON alone, with the same sections for every model: summary, configuration, metrics, data-parallel scaling, baseline,
environment and reproduction, and what the report does not cover.

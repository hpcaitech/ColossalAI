# Benchmark

A shared framework for measuring how fast and how memory-hungry training a model is under a ColossalAI plugin, the
same way for every model, so that reports from different models, plugins and machines can be compared.

Covered so far: throughput, MFU, peak memory, the maximum sequence length and micro-batch on one device, and
data-parallel scaling efficiency. Not covered yet, and listed as such in every report: comparison with a baseline.

## Quick start

```bash
python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16 \
    --micro-batch 2 --out llama-1b.json
```

This trains Llama-3.2-1B (randomly initialized, real size) for 5 warmup and 20 timed steps, twice, each time in fresh
processes, and writes `llama-1b.json` and the rendered `llama-1b.md`. On H20 the summary line reads:

```
12,601 tokens/s on 1 × NVIDIA H20 (12,601 per device), MFU 70.0% of the nominal 148 TFLOPS (73.7% of the measured
140.6), peak reserved memory 48.30 GiB.
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

Find the longest sequence (at micro-batch 1) or the largest micro-batch (at the entry's sequence length, or
`--seq-len`) that trains on one device, then measure there:

```bash
python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16 \
    --search seq_len
python -m tests.kit.benchmark.run --entry transformers_qwen3_for_causal_lm --size 1.7b --config zero2_bf16 \
    --search micro_batch --gradient-checkpointing
```

The search runs on multiples of `--granularity` (256 tokens, 1 sample) up to `--upper` (for a sequence length, the
model's `max_position_embeddings`). See "Maximum on one device" below.

Data-parallel scaling over 1, 2, 4 and 8 devices (weak scaling: each device keeps its micro-batch and sequence
length):

```bash
python -m tests.kit.benchmark.run --entry transformers_llama_for_causal_lm --size 1b --config zero2_bf16 \
    --micro-batch 2 --scaling 1,2,4,8
```

Render a JSON report again, e.g. after collecting it from another machine:

```bash
python -m tests.kit.benchmark.report llama-1b.json > llama-1b.md
```

Run on idle devices. Other processes slow the steps down and count towards the device-side memory; the report records
how much device memory was in use when the run started (this process's own context included, a few hundred MiB on
CUDA), and flags the run when that exceeds 10% of a device. A process that starts on the devices mid-run is not caught
by this; for data-parallel scaling on a shared machine, check `nvidia-smi` before and after.

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

A model whose FLOPs have a closed formula gives it as `flops_fn` (`decoder_flops_per_token` covers Llama-, Qwen3- and
Mistral-style decoders); any other model is counted (see MFU below).

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

Peak memory is tracked from the first step on and is the maximum over ranks.

### Maximum on one device

`search.py` probes one value at a time. A probe trains as many full steps (forward, backward, optimizer step) as the
measurement, in a fresh process. Near the limit the allocator's high-water mark keeps rising as its cache fragments:
sequence lengths of a Llama-3.2-1B that trained 3 steps on H20 ran out of memory later in the 25-step measurement, so a
shorter probe reports a maximum that cannot be measured. A fresh process starts with an empty allocator, so a failed
probe cannot make the next one fail. A probe fits unless it runs out of device memory (or, with
CPU offloading, its process is killed); any other error stops the search.

From the entry's default the search doubles until a probe does not fit (or halves until one does), then bisects
between the largest value that fits and the smallest that does not. The largest fitting value is then measured with
the full protocol; if that runs out of memory after all (near the limit a run is not deterministic to the last token),
the next smaller fitting value is measured and the report says so. The report lists every probe. The allocator settings
(`PYTORCH_CUDA_ALLOC_CONF`, e.g. `expandable_segments:True`) change the result and are recorded in the environment.

### Data-parallel scaling

`scaling.py` measures the same setup on each device count with the full protocol. The efficiency at N devices is the
throughput per device against the smallest count measured (one device, unless the model only fits sharded). Each point
then runs 3 more steps under the profiler, outside the timing, and reads from the device timeline the share of a step
in which communication kernels (NCCL) run, and the share in which they run while no other kernel does: the time the
device waits for communication, which is what an efficiency below 1 usually comes from. The shares are the largest over
ranks. On Ascend the profiler is a different API and the shares are not read yet.

### MFU

MFU = model FLOPs per token × tokens/s per device ÷ the device's peak: the matmuls of one forward and backward (PaLM,
appendix B), without the forward that gradient checkpointing recomputes. Causal attention is halved, as Megatron-LM
and FlashAttention count it: the mask leaves half of the score matrix and fused kernels skip the other half. PaLM and
torchtitan count the whole matrix, which at short sequences gives a few points more MFU than here; at long ones it
overstates the work so much that a Llama-3.2-1B at 20k tokens measured 105% MFU on H20. With gradient checkpointing the
report adds HFU, which counts the recomputed forward too. An MFU above 100% is flagged.

FLOPs per token come from two independent sources (`flops.py`), and an entry with a formula reports both:

- the entry's `flops_fn`, a closed formula from the config;
- `FlopCounterMode` over one forward and backward of the real-size model built on the meta device (as fake tensors:
  no memory, a second or two even for 8B). Calls to `scaled_dot_product_attention` are counted from their shapes,
  halved for a model declared `causal_attention=True` (the counter cannot see a mask's values on the meta device);
  attention written as plain matmuls (`attn_implementation="eager"`) is counted whole, as it is computed whole.

They differ by less than 1e-6 for every declared Llama and Qwen3 size (the counter also sees the rotary embedding's
frequency product); a difference above 5% is flagged, as one of them then does not describe the model.

The peak (`peak.py`) is given twice. The nominal one is the vendor's dense figure, the denominator everyone uses. The
measured one is the best median of large square matmuls on our machines; it shows how much of the gap is the device
itself: a 910B2 is power-capped and clocks down under load, so it reaches about 77% of its nominal bf16 peak even in a
plain matmul. A device not in the table gets no MFU, and the report says so.

| Device | dtype | Nominal | Measured |
| --- | --- | --- | --- |
| H20 | bf16 / fp16 | 148 | 140.6 / 140.5 |
| H20 | fp32 | 44 | 32.5 |
| Ascend 910B2 | bf16 / fp16 | 376 | 290.8 / 260.5 |
| Ascend 910B2 | fp32 | 99 | 75.3 |

TFLOPS, dense; sources are in `peak.py`. All device calls go through
`colossalai.accelerator`, so the same code runs on CUDA and Ascend NPU; on CPU only the host memory is reported.

## Report

`run.py` writes one JSON file per setup (`schema_version` 2): the setup, every timed step of every repeat, the memory
peaks, the FLOPs per token and the peak used, the summary statistics, the environment (device, driver, clock and power
limits, library versions), the repository commit and whether the tree was dirty, and the command that reproduces it.
The Markdown is rendered from the JSON alone, with the same sections for every model: summary, configuration, metrics,
FLOPs and peak, the maximum on one device (when searched), data-parallel scaling (when measured), baseline,
environment and reproduction, and what the report does not cover.

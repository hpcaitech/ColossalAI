# Benchmark

A shared framework for measuring how fast and how memory-hungry training a model is under a ColossalAI plugin, the
same way for every model, so that reports from different models, plugins and machines can be compared.

Covered: throughput, MFU, peak memory, the maximum sequence length and micro-batch on one device, data-parallel
scaling efficiency, and comparison with another setup (a plain-PyTorch FSDP2 baseline, another implementation of the
same model). A report lists whatever it does not cover, so that a missing number is never read as zero.

## Layers

The code is four layers, one package each; a layer only uses the ones below it.

| Layer | Package | Does | Model lines |
| --- | --- | --- | --- |
| 1 | `config/` | what is measured: a model's `BenchmarkSpec`, the plugin configurations, one `Setup` | declare a `BenchmarkSpec` once per model |
| 2 | `runner/` | runs one `Setup` once in fresh processes and records raw data; computes no metric | do not touch |
| 3 | `metrics/` | turns raw records into metrics; pure functions, each metric computed in one place | do not touch; ask for new metrics |
| 4 | `scenarios/` | decides which setups to run (single, maximum search, scaling, comparison), writes the report | run them with `run.py` |

Every scenario changes one field of a `Setup` and measures again: the search changes the sequence length or
micro-batch, scaling the device count, a comparison any one field (the plugin, the implementation).

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

Options: `--devices N` (one process per device), `--seq-len`, `--micro-batch` (per device), `--gradient-checkpointing`,
`--implementation` (one the entry declares), `--warmup-steps`, `--timed-steps`, `--repeats`. `--config` is one of:

| Config | Plugin |
| --- | --- |
| `ddp_fp32`, `ddp_fp16` | `TorchDDPPlugin`, without or with `mixed_precision="fp16"` |
| `zero1_bf16`, `zero2_bf16` | `LowLevelZeroPlugin(stage=1 or 2, precision="bf16")` |
| `zero2_bf16_offload` | the same with `cpu_offload=True` |
| `gemini_bf16` | `GeminiPlugin(precision="bf16")` |
| `fsdp_fp32`, `fsdp_bf16` | `TorchFSDPPlugin`, without or with bf16 `MixedPrecision` |
| `pytorch_fsdp2_bf16` | no ColossalAI: the built-in baseline (see "Comparison" below) |

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

Add `--compare FIELD=VALUE` to any of these to measure the same setup with one field changed as well (for `--scaling`,
at the largest device count; for `--search`, at the maximum found) and compare the two in the report:
`--compare config=pytorch_fsdp2_bf16` against the plain-PyTorch baseline, `--compare implementation=default` a fused
implementation against the model as built.

Render a JSON report again, e.g. after collecting it from another machine:

```bash
python -m tests.kit.benchmark.scenarios.report llama-1b.json > llama-1b.md
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
`config/model.py`. Weights are random: throughput and memory of a dense model do not depend on them.

A model that gets an L4 report gives a FLOPs formula as `flops_fn(model, seq_len)`, in its own model_zoo file next to
the `BenchmarkSpec`: the line knows its layers best (e.g. Gated DeltaNet, a MoE router), and MFU uses the formula.
FLOPs are also counted automatically (see MFU below), but only as a check: the count sees the matmuls the reference
implementation runs, not custom kernels, routing that depends on values, or which positions a mask tensor hides, and
it is verified against formulas only for standard decoders. Every measurement checks the formula against the count and
flags a difference above 5%; `tests/test_benchmark/test_benchmark_device.py` checks every declared formula at its real
sizes in CI. On a difference, check the formula first. If it is right, open a GitHub issue with the model and both
numbers: either the counter has a gap that can be fixed (an attention it reads wrongly, a custom kernel with no
registered FLOPs), or the model cannot be counted on the meta device (routing that depends on values). Such a model is
counted instead in one extra, untimed step on the devices at the end of its measurement, with its real weights
(planned with MoE support); only what neither count can see is checked by hand by the benchmark owners. Until then the report's warning cites the issue; a report with an
unresolved warning is not a final L4 report.
Without a formula, MFU comes from the count alone and is not verified. A formula follows these conventions, so that
MFU compares across lines:

- model FLOPs per trained token: the matrix multiplications of one forward and backward, 2 FLOPs per multiply-add
  forward and twice that backward, i.e. 6 per weight per token (PaLM, appendix B);
- not the forward that gradient checkpointing recomputes (that is HFU, counted separately);
- causal attention halved (its score matrix and the product with the values), non-causal attention whole;
- element-wise operations, norms and softmax left out;
- a MoE layer counts the experts each token is routed to, not all of them. Which experts changes from step to step,
  but each token always goes through its top-k, so the FLOPs per token do not; an implementation that drops tokens
  over an expert's capacity, or pads to it, is written as if every token went through its top-k, and the report says
  what was dropped or padded.

Build on `decoder_flops_per_token` (Llama-, Qwen3- and Mistral-style decoders) for the standard attention and MLP and
add what is specific to the model. Where the count cannot run (a router whose dispatch depends on the values, a layer
with no meta implementation), the formula is the only source and is reviewed by the benchmark owners.

A model that can be computed in several ways (fused kernels, ColossalAI's `ColoAttention` or MoE layer, the plain
PyTorch modules) declares the others as `implementations`: functions that take the model as built (the `"default"`
implementation) and swap its modules. Import ColossalAI inside the function, not at the top of the file: model_zoo
imports every entry, and must not pull in ColossalAI. FLOPs are counted in `reference_implementation` (default
`"default"`), which must run on the meta device; fused kernels and expert-parallel layers do not.

```python
def fused(model):
    from colossalai.shardformer.layer import ColoAttention  # imported only when this implementation runs
    ...  # replace the attention modules in place, or return a new model

benchmark = BenchmarkSpec(models=..., implementations={"fused": fused})
```

## What is measured, and how

Every model goes through the same protocol (`runner/measure.py`):

- **Data**: each rank builds one batch of fixed shape, seeded by its rank, and feeds it to every step, so data loading
  is not measured.
- **Step**: forward, `booster.backward`, `optimizer.step`, `optimizer.zero_grad`. The device is synchronized before
  the step, after the backward and after the step, so each time is the device's work and not only the host launching
  it. The time after the backward splits the step into forward + backward (including the gradient reduction a plugin
  runs in its backward hooks) and the optimizer (including the parameter gathering a sharded optimizer runs in its
  step).
- **Steps**: warmup steps are discarded. The time of a timed step is the maximum over ranks.
- **Repeats**: each repeat runs in fresh processes. Two repeats whose median step times differ by more than 5% are
  flagged in the report: the machine was not stable enough to compare the numbers with anything.

| Metric | Definition |
| --- | --- |
| Throughput | tokens of all ranks per step ÷ median step time; also per device and in samples/s. The median keeps one stalled step from moving the headline; p10 and p90 are reported next to it |
| Step split | median time to the end of the backward ÷ median step time; the optimizer is the rest |
| Peak allocated | `max_memory_allocated`: memory held by tensors |
| Peak reserved | `max_memory_reserved`: memory held by the caching allocator, which is what runs out first |
| Peak on device | peak reserved plus what the process holds outside the allocator (CUDA context, communication buffers), measured at the end of the run: the device-side number `nvidia-smi` or `npu-smi` shows |
| After setup | allocated once the model and optimizer are boosted: weights, master weights and optimizer state before the first step |
| Host RSS peak | peak resident memory of a process, model construction included; what CPU offloading costs |

Peak memory is tracked from the first step on and is the maximum over ranks.

### Maximum on one device

`scenarios/search.py` probes one value at a time. A probe trains as many full steps (forward, backward, optimizer step) as the
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

`scenarios/scaling.py` measures the same setup on each device count with the full protocol. The efficiency at N devices is the
throughput per device against the smallest count measured (one device, unless the model only fits sharded). Each point
then runs 3 more steps under the profiler, outside the timing, and reads from the device timeline the share of a step
in which communication kernels (NCCL) run, and the share in which they run while no other kernel does: the time the
device waits for communication, which is what an efficiency below 1 usually comes from. The shares are the largest over
ranks. On Ascend the profiler is a different API and the shares are not read yet.

### Comparison

`scenarios/compare.py` measures the same setup with one field changed, by the same protocol, and the report puts the
two side by side with their throughput and memory ratios. The usual comparisons are the plugin and the implementation.

The plugin `pytorch_fsdp2_bf16` (`config/plugins.py`) trains the same model without ColossalAI, with PyTorch FSDP2 the way torchtitan sets it up:
`fully_shard` on every transformer block (the classes a HuggingFace model lists in `_no_split_modules`), then on the
root, computing in bf16 and reducing gradients in float32, with `torch.optim.AdamW`. It is timed by the same protocol,
so the comparison is like with like: same model, sequence length, micro-batch, precision, gradient
checkpointing and device count. Results of other frameworks (Megatron-LM, DeepSpeed, a model's own training code) are
measured by other code and are not part of these reports.

### MFU

MFU = model FLOPs per token × tokens/s per device ÷ the device's peak: the matmuls of one forward and backward (PaLM,
appendix B), without the forward that gradient checkpointing recomputes. Causal attention is halved, as Megatron-LM
and FlashAttention count it: the mask leaves half of the score matrix and fused kernels skip the other half. PaLM and
torchtitan count the whole matrix, which at short sequences gives a few points more MFU than here; at long ones it
overstates the work so much that a Llama-3.2-1B at 20k tokens measured 105% MFU on H20. With gradient checkpointing the
report adds HFU, which counts the recomputed forward too. An MFU above 100% is flagged.

FLOPs per token come from two independent sources (`metrics/compute.py`), and an entry with a formula reports both:

- the entry's `flops_fn`, a closed formula from the config;
- `FlopCounterMode` over one forward and backward of the real-size model, in its reference implementation, built on
  the meta device (as fake tensors: no memory, a second or two even for 8B). Calls to `scaled_dot_product_attention` are counted from their shapes,
  halved for a model declared `causal_attention=True` (the counter cannot see a mask's values on the meta device);
  attention written as plain matmuls (`attn_implementation="eager"`) is counted whole, as it is computed whole.

They differ by less than 1e-6 for every declared Llama and Qwen3 size (the counter also sees the rotary embedding's
frequency product); a difference above 5% is flagged, as one of them then does not describe the model.

The peak (`metrics/compute.py`) is given twice. The nominal one is the vendor's dense figure, the denominator everyone uses. The
measured one is the best median of large square matmuls on our machines; it shows how much of the gap is the device
itself: a 910B2 is power-capped and clocks down under load, so it reaches about 77% of its nominal bf16 peak even in a
plain matmul. A device not in the table gets no MFU, and the report says so.

| Device | dtype | Nominal | Measured |
| --- | --- | --- | --- |
| H20 | bf16 / fp16 | 148 | 140.6 / 140.5 |
| H20 | fp32 | 44 | 32.5 |
| Ascend 910B2 | bf16 / fp16 | 376 | 290.8 / 260.5 |
| Ascend 910B2 | fp32 | 99 | 75.3 |

TFLOPS, dense; sources are in `metrics/compute.py`. All device calls go through
`colossalai.accelerator`, so the same code runs on CUDA and Ascend NPU; on CPU only the host memory is reported.

## Code layout

```
tests/kit/benchmark/
├── run.py              command line: one scenario, its JSON report and Markdown
├── config/             layer 1
│   ├── model.py        BenchmarkSpec: a model_zoo entry's real sizes, implementations and how to feed it
│   ├── plugins.py      the plugin configurations, and plain-PyTorch FSDP2, the built-in baseline
│   └── setup.py        Setup: one measurement's conditions
├── runner/             layer 2
│   ├── measure.py      one run in fresh processes: build, boost (or wrap), time, memory; out-of-memory recorded
│   ├── environment.py  versions, device, clock and power limits, device memory in use at the start
│   └── timeline.py     device timeline of profiled steps: steps, communication and compute intervals
├── metrics/            layer 3
│   ├── time.py         step time, throughput, step split
│   ├── compute.py      FLOPs per token (formulas and counting), peaks, MFU / HFU
│   ├── memory.py       memory peaks
│   ├── parallel.py     communication shares, scaling efficiency
│   ├── checks.py       when the numbers must not be used
│   └── summary.py      all metrics of a measurement; scaling curve; comparison ratios
└── scenarios/          layer 4
    ├── single.py       one setup, repeated in fresh processes
    ├── search.py       largest sequence length / micro-batch on one device
    ├── scaling.py      data-parallel scaling over device counts
    ├── compare.py      the same setup with one field changed
    └── report.py       JSON report and the Markdown template
tests/test_benchmark/   self-tests: CPU (test_benchmark_{core,search,scaling,compare}.py) and devices
                        (test_benchmark_device.py)
```

`config/model.py` and `metrics/compute.py` import neither ColossalAI nor model_zoo, as model_zoo entries import them.
The self-tests run in CI (the CPU tests in the CPU batch, the device tests with two GPUs); benchmarking real models is a
tool, not a test.

## Report

`run.py` writes one JSON file per scenario (`schema_version` 3): the measurement (its setup, every raw run with every
timed step of every rank, the FLOPs per token and the peak used, the summary of its metrics), the search, scaling curve
or comparison when there is one, the environment (device, driver, clock and power
limits, library versions), the repository commit and whether the tree was dirty, and the command that reproduces it.
The Markdown is rendered from the JSON alone, with the same sections for every model: summary, configuration, metrics,
FLOPs and peak, the maximum on one device (when searched), data-parallel scaling (when measured), comparison,
environment and reproduction, and what the report does not cover.

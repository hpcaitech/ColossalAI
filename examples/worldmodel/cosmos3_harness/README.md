# Cosmos 3 harness: training contract, benchmarks and evaluation

This is the shared GPU/Ascend harness for Cosmos 3. The DROID action-policy SFT recipe lives next door in
[`../cosmos3`](../cosmos3/README.md) and owns the data pipeline (LeRobot decoding, the VAE, the prompt and the noise
schedule); the framework pieces live in `colossalai/shardformer/`; this directory owns the data contract, the loss
reduction, the training loop and the evaluation/benchmark manifests.

## What is supported

- One packed batch containing several samples, laid out as a global AR block followed by a global DM block.
- BF16, data parallel + ZeRO, with optional `--cp-size N` Ulysses context parallelism (`tp=pp=1`).
- `--attn-backend` (alias `--backend`) chooses attention; `--device` independently selects the accelerator.
  `auto` detects the device and uses its communication backend (NCCL/HCCL).
- Immutable model/optimizer checkpoint generations, one public training state and one RNG file per rank.
- Explicit `--loss-objective droid` or `generic-vision-token-mean`, and explicit `--replay-mode replicated|sharded`.
- A grouped held-out split that never puts two windows of the same trajectory on both sides.

TP/PP, Super/Nano variants and audio generation are outside this entry point's scope.

## Quick checks that need no accelerator or checkpoint

```bash
export PYTHONPATH=.
python examples/worldmodel/cosmos3_harness/train.py --dry-run
bash examples/worldmodel/cosmos3_harness/test_ci.sh
```

`--dry-run` packs two synthetic samples, checks every rewritten index, computes a velocity MSE and a token-normalised
loss. It does not import diffusers and does not initialise a process group. Record the runtime baseline with:

```bash
python examples/worldmodel/cosmos3_harness/environment_report.py \
    --model nvidia/Cosmos3-Edge --revision <fixed revision> \
    --output logs/cosmos3-environment.json
```

## Held-out split

Produce a JSONL where every record has `sample_id` and `trajectory_id`, then:

```python
from examples.worldmodel.cosmos3_harness.heldout import split_by_group, write_split_manifest

train_ids, validation_ids, manifest = split_by_group(records, validation_fraction=0.1, seed=0)
write_split_manifest(manifest, "heldout.json")
```

The manifest stores a fingerprint of both sample lists. Both devices must load the same file; editing one list makes
the fingerprint check fail instead of silently comparing different splits.

## Real run

`--replay-pack` is a `.pt` file holding a list of packs; each pack is a list of sample dictionaries shaped like the
packer output (`und_len`, `sequence_length`, `input_ids`, `text_indexes`, `position_ids`, the vision/action/sound
fields). `--loss-objective generic-vision-token-mean` requires `vision_target` per sample and rejects action/sound
inputs. This legacy generic objective averages vision elements; it is not the DROID training objective.
`--loss-objective droid` instead requires a complete per-sample `targets` mapping from `cosmos3.packing.pack_one`
(`vision`, `action`, `raw_action_dim`, `vision_noisy_mask`, `action_noisy_mask`) and delegates to that recipe's
`compute_loss`. Masks retain full targets and denominators, channels are masked by raw action width, and samples
receive equal weight across the whole DP accumulation window.

```bash
torchrun --nproc_per_node=2 examples/worldmodel/cosmos3_harness/train.py \
    --model /path/to/Cosmos3-Edge \
    --replay-pack /path/to/replay-packs.pt \
    --eval-pack /path/to/heldout-packs.pt \
    --heldout-manifest /path/to/heldout.json \
    --device cuda --attn-backend cuda --zero-stage 2 --steps 100 \
    --replay-mode replicated --loss-objective generic-vision-token-mean \
    --save-dir /path/to/run
```

`--heldout-manifest` verifies the manifest's fingerprint and then checks both pack files against its sample lists:
a training sample the manifest assigns to validation, or an eval sample it never assigned there, is an error
instead of a number quietly measured on a different split. Without an `--eval-pack` only the training side is
checked.

With `--replay-mode replicated`, every rank reads the same replay file; DP peers stride over distinct packs while
CP peers share the same DP coordinate and pack. At least one pack per DP rank is required. With `sharded`, use
`--replay-pack '/path/shard-{dp_rank}.pt'`: each DP rank reads a distinct file, shared by its CP peers. File hashes
are checked across the topology before training. `--gradient-accumulation-steps N` accumulates every microstep;
the cursor counts consumed global pack slots in replicated mode and consumed local slots in sharded mode.
`--checkpoint-mode none|full|selective` also accepts the alias `--activation-checkpointing`.

Resume with `--resume /path/to/run` or an individual `checkpoint-*` generation. `latest.json` is atomically replaced
only after model/optimizer shards, the public `training_state.pt`, and every rank's Python/NumPy/Torch/accelerator
RNG file are written and covered by a SHA256 completion manifest. Incomplete or corrupted generations are rejected
before loading weights. Resume requires matching world size, CP topology, objective, optimizer configuration,
accumulation and replay fingerprints. Old unverified checkpoint layouts are rejected. The same command runs on
Ascend with `--device npu --attn-backend npu`; communication is selected from the accelerator.

## Kernel and benchmark evidence

```bash
python examples/worldmodel/cosmos3_harness/benchmark.py \
    --backend cuda --device cuda --samples 1 2 4 8 \
    --output logs/cosmos3-attention.json
python examples/worldmodel/cosmos3_harness/benchmark.py \
    --backend naive --device cpu --samples 1 2 --iterations 2 --warmup 1 \
    --output /tmp/cosmos3-reference.json
```

The JSON records the environment, shape, backend, p50/p95 latency, K/V interleave cost and the actual fused path. Run
the naive and fused cases with the same tensors and token budget before quoting a ratio. A naive OOM is recorded as an
OOM, not extrapolated.

The optimizer arms and the end-to-end step have their own harnesses. Both report the same `env` / `conditions` /
`results` JSON and take the environment block from `bench_ops.collect_env`, so their numbers are comparable:

```bash
python examples/worldmodel/cosmos3_harness/bench_ops.py --op all --device npu \
    --num-tensors 432 --numel 400000 --output logs/cosmos3-ops.json
torchrun --nproc_per_node=8 examples/worldmodel/cosmos3_harness/train.py \
    --model /path/to/Cosmos3-Edge --replay-pack /path/to/bench-packs.pt --save-dir /tmp/bench \
    --device npu --attn-backend npu --precision bf16 --zero-stage 2 --steps 40 --bench-warmup 10 \
    --replay-mode replicated --loss-objective generic-vision-token-mean \
    --bench-json logs/cosmos3-step.json
```

`bench_ops.py` measures `plain` / `foreach` / `fused` AdamW and `HybridAdam` on a synthetic parameter set of the
chosen shard geometry, plus the unscale/clip path before and after the multi-tensor change. `--bench-json` times the
real step and drops the first `--bench-warmup` steps. A change is accepted only when the median and the minimum both
improve; the per-step samples are kept in the JSON so a bimodal run is visible.

## Reasoner evaluation

```bash
python examples/worldmodel/cosmos3_harness/evaluate_reasoner.py \
    --benchmarks MMBench-Dev HallusionBench \
    --model-revision <pinned revision> \
    --manifest logs/cosmos3-reasoner-manifest.json
```

The manifest pins the VLMEvalKit dataset name, split, language, decoding and scorer. The actual evaluation runs in the
official VLMEvalKit environment; this directory only makes the GPU and Ascend runs share one manifest and one
per-sample response format. A subset of the General category is not a reproduction of `General Avg 59.7`.

## DROID closed-loop evaluation with RoboLab

[`ROBOLAB.md`](ROBOLAB.md) documents the shared GPU/NPU policy server, pinned evaluation manifest,
official RoboLab runner, strict episode validation and paired comparison. Isaac simulation runs on an RTX GPU;
the policy can run on a separate CUDA or Ascend host. The success metric comes from official RoboLab task
predicates, independently of the DROID held-out training loss.

## Testing

`test_ci.sh` runs the CPU-safe checks: packing/index rewriting, held-out fingerprinting, the differentiable loss
reduction, the benchmark reference, the evaluation manifest and the CPU two-way tests. GPU/NPU tests are registered in
`tests/test_shardformer/` and run through the repository's batch scripts.

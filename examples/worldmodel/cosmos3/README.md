# Cosmos 3 DROID Vision+Action SFT

This example follows `action_policy_droid_nano` with the diffusers Cosmos 3 model and ColossalAI two-way attention. It trains the generation tower and action heads while freezing the AR tower. It supports Ascend and CUDA; the accelerator chooses the device and communication backend, independently of the attention and optimizer choices.

## Alignment reference

The integration uses the numerical/data contracts merged into official `npu/dev` at `7c8bd8af36ef5f095c855193ae77c2366c5865f5` (2026-11-12): [6516](https://github.com/hpcaitech/ColossalAI/pull/6516), [6518](https://github.com/hpcaitech/ColossalAI/pull/6518), [6519](https://github.com/hpcaitech/ColossalAI/pull/6519), [6522](https://github.com/hpcaitech/ColossalAI/pull/6522), [6525](https://github.com/hpcaitech/ColossalAI/pull/6525) and the follow-up [6526](https://github.com/hpcaitech/ColossalAI/pull/6526). This branch syncs that state and keeps the local CP, HSDP, selective checkpointing, VAE and replay extensions on top; where the two implement the same contract, the official data/precision semantics are unchanged. The synced reference recipe is under [upstream](upstream/README.md).

The default contract is:

- Shared torchvision v2 crop, resize and color jitter across all three views, followed by camera composition, uint8 truncation, capped canvas resize and reflection padding.
- Gripper inversion for both the action and initial state, random annotation selection, full JSON prompt metadata, EOS and generation markers.
- Online VAE encode in BF16, checkpoint channel normalization, latent-space removal of canvas padding, then FP32 noise, sigma, interpolation and targets. Only the model payload is cast to BF16.
- FP32 fractional mRoPE positions and rotary constants. The action temporal offset is zero with initial state and one without. Patch grids round up for odd latent sides; padded noisy action channels are zero.
- FP32 timestep MLP arithmetic and unrounded sinusoidal features, while trainable parameter storage stays BF16. Plugin-wide input and buffer casts default off for this entry.
- The DROID loss is a weighted sum of modality means averaged per sample. The noisy mask removes conditioning errors while conditioning elements remain in the denominator. Unequal-length samples distinguish this objective from a global supervised-element mean.

The upstream augmentation, collate, packed layout, position and target regression assertions are preserved. The former arbitrary `--latents` route has been removed: a latent file without complete normalization, geometry and augmentation provenance cannot stand in for online DROID training. Complete replay packs have their own entry in [cosmos3_harness](../cosmos3_harness/README.md).

## Environment

Install ColossalAI in the accelerator's supported PyTorch environment, then install this directory's [requirements.txt](requirements.txt). The value-transforming dependencies are pinned, including diffusers 0.40.0 and PyAV 18.1.0. Install torchvision matched to that environment's torch/torch_npu build. Run manifests record the resolved versions.

The repository-wide diffusers 0.35.2 pin conflicts with this example, whose model classes require 0.40.0. This example does not change the repository-wide dependency contract. Transformers follows the repository's 5.17.0 pin. CUDA attention additionally needs a PyTorch CUDA build supplying `torch.nn.attention.varlen`; requesting it without the adapter raises an error.

A checkpoint directory must contain `transformer/`, `vae/` and `text_tokenizer/`. The dataset is DROID in LeRobot v3 layout. A parent containing both `success/` and `failure/` includes both roots; `--success-only` limits it to success. The episode split excludes validation episodes. Supply the recipe's `keep_ranges_1_0_1.json` through `--keep-ranges` to use the curated windows; omitting it trains all eligible windows and is recorded in the manifest.

## Run

From this directory:

```bash
python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
    --data /path/to/Cosmos3-DROID \
    --checkpoint /path/to/Cosmos3-Edge \
    --keep-ranges /path/to/keep_ranges_1_0_1.json \
    --out runs/cosmos3-sft
```

Defaults are 32 samples per independent DP pack and 32 microsteps per optimizer step, with LR `2e-4`, action-head LR multiplied by five, betas `(0.9, 0.99)`, epsilon `1e-8`, weight decay `0.05` and gradient clipping at `1.0`. VAE sample microbatching (`--vae-encode-batch`, for example 4) limits encode-call memory without changing packing or loss.

Logs and metrics report `loss/vision` and `loss/action` in the upstream convention: the masked squared error uses
the full element count as its denominator, averaged per sample, with the recipe's fixed modality weight removed.
This makes these two values directly comparable with upstream flow-matching references. Separate
`loss/vision_supervised_mse` and `loss/action_supervised_mse` metrics renormalize by the supervised element count;
conditioning frames/states and padded action channels do not enter those denominators. `loss/total` and
`loss/objective` both hold the weighted scalar used for backward. For example, an old weighted vision value `1.7`
with 9 latent frames maps to upstream vision loss `1.7 / 10 = 0.17`; its supervised-element MSE is
`0.17 * 9 / 8 = 0.19125`. Historical records are converted on read, without rewriting their JSONL files. This
reporting change does not change the training objective or gradients. The direct stdout line is explicitly rank
0's local pack; aggregated lines show the reporting rank coverage.

The base recipe reinitializes the two action projections and modality embedding. When starting from a trained policy checkpoint, pass `--no-reinit-action-head`. For a pinned comparison, `--action-head-from` imports a complete reference head export or an explicitly named domain slice after initialization.

A single-device smoke run:

```bash
python -m torch.distributed.run --nproc_per_node=1 train_sft.py \
    --data /path/to/Cosmos3-DROID/success \
    --checkpoint /path/to/Cosmos3-Edge-Policy-DROID \
    --no-reinit-action-head --zero-stage 0 \
    --samples-per-rank 1 --grad-accum 1 --max-iter 10 --max-episodes 8
```

`--attn-backend auto` resolves to `npu`, `cuda` or the CPU reference. `--backend` is a compatible alias. `--optimizer hybrid-adam` preserves the FP32-master recipe optimizer; `--optimizer adamw --adamw-mode plain|foreach|fused|auto` selects an explicit comparison. Fused NPU AdamW rejects unsupported devices and missing implementations instead of falling back silently.

## Parallel and checkpoint contracts

`--sp-size` enables Ulysses CP. Only each CP group's leader starts decoding workers, builds the VAE, and draws augmentation, caption dropout and noise. It broadcasts the complete inputs, targets, sample IDs and per-sample AR/DM boundaries. `--cp-debug-identity` hashes these before forward. Independent DP groups use distinct sample partitions; the sampler drops incomplete tails instead of padding them with duplicate examples.

Effective global batch is:

` samples_per_rank × grad_accum × independent_DP_replicas `

CP copies do not multiply it. Counts include every microstep and are reported only once per CP group. The loader cycles across epochs until `--max-iter` optimizer steps have completed. Metrics separate synchronized end-to-end step time from host dispatch time, label device-event/fallback section timings, and record process-group ranks and actual hostnames.

`--hsdp-shard-size` partitions the mixed DP group into logical shard and replica groups. Contiguous ranks are on one host only if the launcher places them there; the manifest makes this observable. HSDP needs ZeRO stage 1 or 2. CPU tests cover four-rank AdamW, accumulation, nonzero gradient clipping, state round trips and async all-gather; actual accelerator streams/offload transfers and multi-host NCCL/HCCL remain device validation work.

`--activation-checkpointing full|selective|none` (`--checkpoint-mode` alias) selects one mode. Full and selective cannot be stacked. `--selective-budget-gib` is an estimate of saved activations, re-evaluated when pack geometry/dtype changes, not a hard HBM limit. Statistics expose the chosen mode, estimates, actual attention dispatch hits and saves. `--dit-sum-mean-norm` rewrites `Cosmos3NemotronRMSNorm`'s `pow(2).mean(-1)` to `square().sum(-1)/H` on NPU (bitwise exact). `--dit-norm-kernel` (off by default) patches `Cosmos3NemotronRMSNorm` on NPU after boost to run `torch_npu.npu_rms_norm` in the forward with a reference-exact autograd backward (~1.2 ppm 1-ulp forward difference, bit-identical backward). `--compile-blocks` compiles only after booster replacements, attention installation, precision wrappers, norm patching and checkpoint configuration. Compiler `auto` resolves to Inductor on CUDA or npugraph_ex on NPU.

For host scheduling and CPU data pipeline tuning on multi-socket NPU nodes:

- **Socket NUMA pinning**: export `COSMOS3_NUMA_PIN=$(python -m examples.worldmodel.cosmos3.numa_affinity --print-map)` so `train_sft.py` pins each rank (and its spawned DataLoader workers) to its NPU's physical socket (48 cores) before allocations, eliminating cross-socket stragglers; unset is a no-op. In a heterogeneous job prefer `COSMOS3_NUMA_PIN=auto`, which derives each node's own map in-process (node5/node9 have 4 domains, the rest 8); a failed derivation logs a note and continues unpinned.
- **Threads**: combine socket pinning with `OMP_NUM_THREADS=6` and `--num-workers 4 --worker-threads 4` (`--worker-threads` restores intra-op threads inside PyTorch workers, which `OMP_NUM_THREADS` cannot reach). `COSMOS3_FAST_JITTER=1` compiles `ColorJitter`'s hue pass via Inductor (set `TORCHINDUCTOR_CACHE_DIR` on local disk). `--prefetch-factor` defaults to `0` (PyTorch's default 2; deeper queues are exposed only for diagnostics).

`--save-every` writes the same immutable, checksummed generations the harness uses, under `<out>/checkpoints/`: model/optimizer shards, one RNG file per rank, the rank-local noise generator, and a public `training_state.pt` with step, epoch and sampler cursor. `latest.json` is replaced only after the SHA256 completion manifest is written, so an incomplete generation is never resumed. `--resume <run dir or checkpoint-*>` continues the same trajectory: the sampler skips the batches that step already consumed instead of re-decoding them, and the resume is rejected unless the topology, batching, data and seed fingerprint matches.

`--async-save` stages the shards into pinned host buffers and hands them to a background writer; rank 0 publishes the generation from a first background thread and every rank confirms it at the next checkpoint entry or at the end of the run. The step therefore pays the device-to-host copy only — serialization, hashing and the rename overlap the following steps. It needs no TensorNVMe: when that extension is absent the same contract is served by a background thread (`colossalai.utils.safetensors.ThreadedSafetensorsWriter`). `--no-async-save` keeps the previous fully synchronous behaviour, matching the upstream Cosmos recipe default.

Online augmentation replays exactly. Each visit seeds the transforms and the caption draw from `(seed, epoch, index)` (`runtime.VisitRandomness`), so the draw is a property of the visit rather than of which DataLoader worker fetched it; the sampler publishes the epoch through a shared value the workers read. `--num-workers` therefore no longer changes what a resumed run produces, and the streams are restored around each visit so the rest of the step's randomness is untouched. The draws differ from the pre-seeding behaviour (the global worker RNG used to decide), which is the point: two runs with the same seed and the same data now augment identically.

Multi-node runs keep `--out` on each node's local disk, and resume handles that layout. Rank 0's node publishes the generation (model/optimizer/`training_state.pt`); every other node keeps only the `rng-<rank>.pt` and `master-<rank>.pt` files its ranks wrote, and `complete.json` covers all of them because each rank gathers its own digests before the save barrier. On `--resume`, rank 0 shares the generation name and manifest, each rank says whether its node is complete, and a node missing the shared files receives them over the training interconnect (one stream per file, `COSMOS3_CHECKPOINT_TRANSFER_CHUNK_MIB` chunks, written by each node's `LOCAL_RANK` 0) before validation and loading. The node then keeps a published copy, so its next resume is local. `scripts/check_checkpoint_complete.py <run-dir> --world 48` reports whether one node can take part.

## Experimental VAE options

BF16, the diffusers encoder and its default window are the baseline. FP16, fused norm, fast convolution, wider chunk windows and compilation are independent opt-ins:

- `--vae-dtype float16`, `--fuse-vae-norm`, `--vae-norm-kernel`, `--fast-vae-encoder`.
- `COSMOS3_VIDEO_NORM_KERNEL=1` enables the TileLang uint8 device normalization kernel (`video_pad_norm.py`) when `--video-uint8` is active; `--vae-norm-kernel` runs the TileLang channel-first RMSNorm kernel on NPU (bitwise exact).
- `COSMOS3_EMPTY_CACHE_GIB` sets the free-memory floor for an `empty_cache()` cleanup, evaluated once per optimizer step; its default is `8` with `--vae-norm-kernel` and `0` (off) otherwise. The decision is all-reduced, so when one rank is below the floor every rank empties its allocator together.
- Both TileLang paths take a file lock while they compile, so the first step's identical shapes are not built and dlopened by every rank at once. The lock lives in `COSMOS3_TILELANG_LOCK_DIR`, else `TILELANG_CACHE_DIR`, else `~/.tilelang`; keep it on storage shared by the ranks and nodes that share the TileLang cache.
- `--vae-chunk-frames N` for a positive multiple of four; zero leaves the diffusers loop intact.
- `--vae-compile jit|aot`; the legacy `--compile-vae` and `--compile-vae-aot` aliases remain. JIT and AOT are mutually exclusive. AOT requires CUDA and chooses a four-frame chunk when none was specified.

The RMSNorm path preserves FP32 normalization before casting and affine operations; it does not clamp an FP16 reciprocal. Chunk tests cover first frames, tails and repeated calls. CPU equivalence does not imply identical output on every accelerator.

AOT uses private runner/rank/process package paths, negative-caches compile/gate/runtime failures and keys variants by input/cache layouts, dtype, device, weight identity/version and compiler settings. A correctness gate checks latent and every returned cache, finite values and structure, per-element error and independent relative L2. Default absolute tolerance scales by each reference tensor's RMS and dtype epsilon; the default L2 budget is twice dtype epsilon. `--vae-gate-rtol`, `--vae-gate-atol` and `--vae-gate-relative-l2` expose the budgets. They still need calibration on real checkpoints. `.data` or external-storage weight changes require `runner.invalidate()`.

Historical speed figures measured before the alignment fixes or with the old loose AOT gate do not establish the speed or correctness of this version. No new GPU/NPU performance claim is made here.

## Monitoring

`train_sft.py` writes per-rank JSONL metrics, a run manifest, and (on rank 0) one aggregated record per log step. The optional [`cosmos3_monitor`](../cosmos3_monitor/README.md) package turns a directory of those run folders into a dashboard with live charts, the full launch/config record, node × card topology, a checkpoint ledger, dashboard-only alerts, and stop/resume buttons.

Enable the lightweight hooks explicitly when you want the extra files:

```bash
bash examples/worldmodel/cosmos3/scripts/train_with_monitor.sh \
    --run-dir /path/to/run --interval 5 -- \
    python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
        --data /path/to/Cosmos3-DROID --checkpoint /path/to/Cosmos3-Edge \
        --out /path/to/run --log-path /path/to/train.log
```

Then serve the dashboard from the node or a workstation:

```bash
python -m examples.worldmodel.cosmos3_monitor serve --roots /path/to/runs --port 18790
```

When every node keeps `--out` on its own local disk, rank 0 can read only its own rank files. Start the
dashboard on a host that can SSH to all nodes and mirror them instead:

```bash
python -m examples.worldmodel.cosmos3_monitor mirror \
    --cache-root ~/.cache/cosmos3_monitor/mirror \
    --nodes node1,node5,node6,node7,node8,node9 \
    --run-id <run> --out <out> --metrics-dir <out>/metrics \
    --agent-path <shared-path>/remote_agent.py \
    --remote-prefix 'docker exec nzf-cai-dev' --interval 5
python -m examples.worldmodel.cosmos3_monitor serve --roots ~/.cache/cosmos3_monitor/mirror --port 18790
```

The mirror merges every node's rank, heartbeat, terminal and system files; the dashboard then reports all 48
ranks, per-node coverage and a node with no heartbeat. See the monitor README for the full option list.

The hooks do not add work on non-log steps, except the stop agreement described next. Use `--no-heartbeat`, `--no-checkpoint-ledger` and `--no-aggregate-metrics` for a benchmark that must be free of them, and `COSMOS3_MONITOR_COLLECT=0` to skip the per-node system collector.

`--graceful-stop` (default on) lets the dashboard stop the run after its current step and resume it later. A watcher thread reads `<out>/control/request.json`; when the loop agrees, it writes one more generation for exactly that step and exits as `stopped` (not `failed`), so `--resume` continues from there. The agreement is one four-byte all-reduce per step — the only per-step collective this entry adds — because every rank must leave on the same iteration. `--no-graceful-stop` removes it and the dashboard then reports the run as not stoppable; `--stop-poll-every N` trades a later stop for fewer collectives and `--control-poll-seconds` tunes the watcher. To resume from the dashboard, start the server with `--launch-template` (see the monitor README); otherwise it shows the exact `--resume` argument to add by hand.

## Validation

Run from the repository root:

```bash
python -m pytest -q -o addopts='' examples/worldmodel/cosmos3/tests examples/worldmodel/cosmos3_harness/tests tests/test_booster/test_plugin/test_hybrid_casting.py tests/test_shardformer/test_model/test_cosmos3_policy.py tests/test_shardformer/test_model/test_cosmos3_fp32_timestep.py tests/test_shardformer/test_model/test_cosmos3_fp32_sinusoid.py tests/test_shardformer/test_model/test_cosmos3_runtime_contracts.py tests/test_shardformer/test_model/test_cosmos3_selective_checkpointing.py tests/test_shardformer/test_model/test_cosmos3_hsdp_optimizer.py tests/test_shardformer/test_model/test_hsdp_and_context_parallel.py tests/test_shardformer/test_model/test_cosmos3_example_helpers.py tests/test_shardformer/test_layer/test_two_way_attn.py tests/test_shardformer/test_model/test_cosmos_mot_packing.py tests/test_shardformer/test_model/test_shard_cosmos3.py
```

CPU coverage includes data/targets, whole-window SFT updates, unequal-length loss/gradient oracles, all input/buffer cast combinations, FP32 timestep wrappers under fullgraph AOT-eager, real Gloo CP contracts, HSDP AdamW and uninterrupted-versus-resumed replay. VAE microbatch 0/1/2 compares inputs, FP32 targets, metadata, RNG, loss and gradients; tiny random Wan encoders cover FP16/BF16/FP32 causal caches. CUDA-only and optional real-checkpoint tests skip without their prerequisites.

The local verification environment is Python 3.11, torch 2.15.0, diffusers 0.40.0 and transformers 5.22.1. The torch/transformers versions differ from the repository's declared training environment. CPU results establish regression contracts, not full numerical alignment on CUDA/NPU. Real production checkpoints, target compiler backends, real AOT packages, multi-host training and loss curves still require device validation.

For a real checkpoint conditioning check:

```bash
python examples/worldmodel/cosmos3/scripts/smoke_conditioning.py --data /path/to/data --checkpoint /path/to/checkpoint --device cuda:0
```

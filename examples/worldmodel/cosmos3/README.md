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

`--activation-checkpointing full|selective|none` (`--checkpoint-mode` alias) selects one mode. Full and selective cannot be stacked. `--selective-budget-gib` is an estimate of saved activations, re-evaluated when pack geometry/dtype changes, not a hard HBM limit. Statistics expose the chosen mode, estimates, actual attention dispatch hits and saves. `--compile-blocks` compiles only after booster replacements, attention installation, precision wrappers and checkpoint configuration. Compiler `auto` resolves to Inductor on CUDA or npugraph_ex on NPU.

`--save-every` writes model exports. They do not include optimizer, RNG and sampler progress and cannot resume the same training trajectory. The harness checkpoint format instead publishes immutable, checksummed generations with a single shared-state writer and separate rank RNG files; see its README for the resume CLI.

## Experimental VAE options

BF16, the diffusers encoder and its default window are the baseline. FP16, fused norm, fast convolution, wider chunk windows and compilation are independent opt-ins:

- `--vae-dtype float16`, `--fuse-vae-norm`, `--fast-vae-encoder`.
- `--vae-chunk-frames N` for a positive multiple of four; zero leaves the diffusers loop intact.
- `--vae-compile jit|aot`; the legacy `--compile-vae` and `--compile-vae-aot` aliases remain. JIT and AOT are mutually exclusive. AOT requires CUDA and chooses a four-frame chunk when none was specified.

The RMSNorm path preserves FP32 normalization before casting and affine operations; it does not clamp an FP16 reciprocal. Chunk tests cover first frames, tails and repeated calls. CPU equivalence does not imply identical output on every accelerator.

AOT uses private runner/rank/process package paths, negative-caches compile/gate/runtime failures and keys variants by input/cache layouts, dtype, device, weight identity/version and compiler settings. A correctness gate checks latent and every returned cache, finite values and structure, per-element error and independent relative L2. Default absolute tolerance scales by each reference tensor's RMS and dtype epsilon; the default L2 budget is twice dtype epsilon. `--vae-gate-rtol`, `--vae-gate-atol` and `--vae-gate-relative-l2` expose the budgets. They still need calibration on real checkpoints. `.data` or external-storage weight changes require `runner.invalidate()`.

Historical speed figures measured before the alignment fixes or with the old loose AOT gate do not establish the speed or correctness of this version. No new GPU/NPU performance claim is made here.

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

# DROID policy evaluation with RoboLab

This bridge runs the **official RoboLab Cosmos3 client**, DROID absolute joint-position controller,
camera preprocessing, 32-step action cache, task termination and success predicates. The policy service loads
a consolidated **diffusers** DROID WAM checkpoint and installs the ColossalAI attention processor used by
the training harness (`cuda`, `npu`, or `naive`). It uses cosmos-framework's `ActionTransformPipeline` and
`FlowUniPCMultistepScheduler` directly.

The Cosmos 3 technical report, §4.2.5 and §6.2.5/Table 19, evaluates RoboLab-120 with vague, default and specific
instructions and 10 rollouts per task. Its policy predicts 32 absolute joint actions at 15 Hz, with 4 denoising
steps, shift 5 and CFG 3. This adapter fixes those generation settings and uses one initial state row, domain 8,
8 raw action channels, and a wrist/left/right 540×640 input canvas. DROID and RoboLab gripper conventions
are inverted on both sides of inference. The official client applies the final `> 0.5` gripper threshold.
The official transform selects a 544×736 model canvas for the `480` tier; this is distinct from the camera canvas.

Sources inspected for this integration:

- [RoboLab Cosmos3 client and runner](https://github.com/NVlabs/RoboLab/tree/9cce00cadd5dc575fd2b8ea3836f5bdcde574229/policies/cosmos3).
- [Cosmos reference RoboLab server](https://github.com/NVIDIA/cosmos-framework/blob/7306dc9c2f99ffecc065006c9813482f70f030e9/cosmos_framework/scripts/action_policy_server_robolab.py).
- [DROID recipe](https://github.com/NVIDIA/cosmos-framework/blob/7306dc9c2f99ffecc065006c9813482f70f030e9/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py).

## Environments

Use separate environments for simulation and policy inference. Install RoboLab and its assets using its
[official setup instructions](https://github.com/NVlabs/RoboLab#installation). Isaac needs Linux and an NVIDIA
RTX GPU even when the policy is served from an Ascend host. Isaac Sim 5.0 / Isaac Lab 2.2.0 is the default stack;
the plan also supports the official `isaac51` and `isaac60` stacks. Keep the exact installed simulator versions
identical for GPU/NPU comparisons: changing PhysX changes contact dynamics.

In the existing ColossalAI policy environment, with matching torch/torchvision and torch_npu already installed:

```bash
pip install -r examples/worldmodel/cosmos3_harness/requirements_robolab.txt
```

The policy also needs a clean cosmos-framework source checkout at the revision in the manifest. Install any
further dependencies required by that pinned checkout in the policy environment. It supplies preprocessing and
the scheduler; this service does not instantiate the native CUDA-only `OmniMoTModel`. The simulator environment
needs `numpy`, `msgpack` and `websockets>=14` for the harness handshake.

The checkpoint directory must contain `transformer/`, `vae/` and `text_tokenizer/`. A native cosmos-framework
safetensors export and a diffusers export use different layouts: convert a native checkpoint with the official
`cosmos_framework.scripts.convert_model_to_diffusers` tool first. For SFT weights, pass
`--transformer-path /path/to/consolidated-diffusers-transformer` alongside the original VAE/tokenizer checkpoint.
Consolidate/export ZeRO shards before serving them.

## One manifest for both devices

Run from ColossalAI with `PYTHONPATH=.`. Planning only needs Python's standard library. Both source checkouts
must be clean. The manifest contains Git commits, selected task source hashes, all three instruction texts,
difficulty labels, simulator stack, episode geometry, seeds and sampling parameters, with a SHA256 fingerprint.
Filesystem roots and devices are runtime information, so copy the **same JSON file** to all hosts.

```bash
export PYTHONPATH=.
python -m examples.worldmodel.cosmos3_harness.robolab plan \
  --robolab-root /path/to/RoboLab \
  --cosmos-root /path/to/cosmos-framework \
  --model nvidia/Cosmos3-Edge-Policy-DROID \
  --model-revision <immutable-checkpoint-revision-or-local-digest> \
  --manifest logs/droid-robolab.json
```

Use `--tasks BananaInBowlTask --instruction-types default --num-envs 1` for a smoke run. Task subsets are labelled
`RoboLab-subset`. The full plan discovers the 120 benchmark tasks from the official metadata and runs 10 parallel
environments × 1 run for each instruction level (3600 episodes). If memory requires fewer environments, use
`--num-envs 2 --num-runs 5` and share that new manifest across both devices; episode geometries cannot be mixed.

Generation uses the same fixed `--policy-seed` for every request, matching the reference server's
`--deterministic-seed` option. This makes random inputs independent of request arrival order. It is a recorded
evaluation choice, not evidence of the report's unreported seed protocol. Initial noise matches the reference's
NumPy `RandomState` FP32 draws, with the seed restarted per modality on CPU before transfer to either device.
The VAE runs in FP32 with HF32/TF32 off.
CFG branches run serially on one device; CFG parallelism is not implemented here. Predicted video stays latent.

## Start the policy service

On the policy host:

```bash
python -m examples.worldmodel.cosmos3_harness.serve_robolab \
  --manifest logs/droid-robolab.json \
  --checkpoint /path/to/Cosmos3-Edge-Policy-DROID-diffusers \
  --cosmos-root /path/to/cosmos-framework \
  --backend npu --device npu:0 --host 0.0.0.0 --port 8000
```

For GPU use `--backend cuda --device cuda:0`; `--backend naive` selects reference attention. The service sends a
binary metadata handshake with the manifest fingerprint, backend and dependency versions, accepts official OpenPI
MsgPack/NumPy observations, and returns `{"action": float32[32,8]}`. Inference is serialized across connections;
each request has fresh sampler history. Errors are returned explicitly to the client. Optional Bearer
authentication reads `COSMOS3_API_TOKEN` on both hosts without storing it in manifests or command lines.
Startup hashes the transformer, VAE and tokenizer files once; this content fingerprint follows the actual SFT
override and stays stable across host paths and Hugging Face cache symlinks.

## Run on the simulator host

```bash
python -m examples.worldmodel.cosmos3_harness.robolab run \
  --manifest logs/droid-robolab.json \
  --robolab-root /path/to/RoboLab \
  --python /path/to/RoboLab/.venv/bin/python \
  --remote-uri ws://npu-host:8000 \
  --backend npu --run-name cosmos3-npu
```

`--dry-run` prints argv arrays without connecting or starting Isaac. A real run checks source revisions, installed
Isaac package versions, server fingerprint and backend before invoking `policies/cosmos3/run.py`. It selects one
instruction level per subprocess and pins renderer, seed and trial counts. Outputs live in
`RoboLab/output/cosmos3-npu/{vague,default,specific}`. Each folder gets `harness_run.json` with commands, service
metadata and actual simulator versions. Reusing a run name resumes through RoboLab's completed-run detection
when its receipt matches exactly. Use `--video-mode none` to disable recording, consistently across both runs.

The native official policy server can also be used with `--external-server`. Configure its checkpoint,
`--format-prompt-as-json`, sampling settings, seed and `--deterministic-seed` to match the manifest yourself.
Its empty handshake cannot verify those settings; this exception is recorded in the receipt. Harness servers
always have their fingerprint checked, including when this flag is present. External results can be summarized,
but cannot pass the verified GPU/NPU comparison without policy provenance.

## Score and compare

A successful run writes `RoboLab/output/cosmos3-npu/summary.json`. Recompute or compare results independently:

```bash
python -m examples.worldmodel.cosmos3_harness.robolab summarize \
  --manifest logs/droid-robolab.json \
  --results-root /path/to/RoboLab/output/cosmos3-npu \
  --output logs/robolab-npu.json
python -m examples.worldmodel.cosmos3_harness.robolab compare \
  --left logs/robolab-gpu.json --right logs/robolab-npu.json \
  --output logs/robolab-comparison.json
```

Summary reads official `episode_results.jsonl` (or legacy JSON), checks task/environment, prompt text, instruction
level, 15 Hz timing, run/env/episode identity, duplicates and expected coverage. It scores the boolean `success`,
independently of the partial subtask `score`. Rates are fractions in [0,1], grouped by instruction, task and official
simple/moderate/complex labels. Task means and episode means agree for a complete balanced run. Raw result hashes
and individual outcomes remain in the summary. Comparison requires the same manifest, full coverage, verified
receipts for every instruction level, identical checkpoint contents and installed simulator versions. It reports
success-rate differences and paired success/failure counts.

`summarize --allow-partial` permits inspection of interrupted runs or imported official results without receipts.
Missing episodes remain explicit. Such results cannot be compared until provenance and complete coverage are
present. Invalid/truncated JSON and malformed records always fail, including in partial mode.

A passing CPU test is not a measured simulation success rate or a reproduction of the Nano checkpoint's Table 19
results by an Edge model.

## Validation

```bash
PYTHONPATH=. pytest -q examples/worldmodel/cosmos3_harness/tests/test_robolab.py
COSMOS3_FRAMEWORK_ROOT=/path/to/cosmos-framework PYTHONPATH=. \
  pytest -q examples/worldmodel/cosmos3_harness/tests/test_robolab.py
```

Tests cover manifests, launch arguments, corrupt/partial result rejection, paired scoring, gripper conversion,
conditioning preservation and authenticated WebSocket exchanges on CPU. The second command additionally checks
the official preprocessing, initial noise, mRoPE arithmetic and UniPC solver. A tiny real diffusers transformer
also checks the packed forward inputs. `test_ci.sh` includes the CPU tests. Hardware
validation requires the checkpoint, an Isaac simulator host and each policy device.

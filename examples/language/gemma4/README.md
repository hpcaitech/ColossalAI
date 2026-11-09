# Gemma 4 dense full text training

This entry point trains **all text parameters**, including PLE, of the original multimodal checkpoint. Vision and
audio parameters are explicitly frozen. It is not multimodal full-parameter training or a convergence benchmark.
Both E2B and E4B use native Transformers forwards; the ShardFormer policy supports data parallelism with TP=1 only.
TP/PP/SP and fused replacements are rejected. This preserves cross-layer KV sharing, PLE and tied embeddings.

Use the maintained Torch 2.13 / Transformers 5.17 image, install this exact source checkout without changing its
dependencies, and expose exactly one allocated H20. The shared model directory is read only. Check actual compute
processes and the team's resource rules before running. Put environments, extensions, checkpoints and logs in your
personal directory. Available host memory is not a guarantee of a container's memory allowance.

```bash
export TORCH_EXTENSIONS_DIR=/your/personal/cache/torch_extensions
MODEL=/mnt/beegfs/ColossalAI/shared-cache/hub/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a
torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 \
  examples/language/gemma4/train.py \
  --model "$MODEL" --output /your/personal/results/e4b-run-001 \
  --steps 3 --sequence-length 128
```

Set device visibility when allocating the container/process; the program requires exactly one visible device.
The output directory must be new. The script requires clean committed source and records commit/import path,
versions, parameters, every step's loss, finite text gradients, CPU FP32 master placement, PLE update samples,
process peak RSS and accelerator peak allocation. LowLevelZero stage 1 uses CPU optimizer offload and FP32 masters.
The reference optimizer is `torch.optim.AdamW(foreach=False)`, avoiding an unverified fused extension for the first run.
The input is deterministic tokenized raw text; no chat template or additional dataset download is needed.

`--save` additionally saves the model and optimizer separately. A successful save does not assert HF reload or exact
resume of the real checkpoint. A first training step allocates optimizer states; measuring only model load/forward
is insufficient for a capacity claim. The tiny resume tests below explicitly check FP32 master preservation.

## Tests

```bash
# No GPU: native TP=1 policy, PLE/shared-KV gradients, non-reentrant checkpointing,
# FP64/FP32 baseline, three-step update/curve equivalence and exact HF reload.
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_model_zoo/test_gemma4.py
# Requires two allocated devices. Includes one- and two-device cases, both tiny profiles,
# DDP FP32/BF16 autocast, ZeRO-1/2 BF16+offload, Gemini/FSDP BF16 and lazy/checkpoint combinations.
python -m pytest -q tests/test_booster/test_plugin/test_gemma4.py
# E2B: 1/2-device numerical alignment, token-weighted accumulation, exact resume and HF export.
python -m pytest -q tests/test_booster/test_plugin/test_gemma4_alignment.py
```

The GPU matrix is in batch 5 (excluded from batch 8 to avoid duplication); CPU checks are in both CPU entry lists.
CPU/Gloo checks also compare 1/2-process DDP with single-process gradient accumulation, retaining unequal valid-token
counts, and compare one-step weights. GPU alignment tests compare logits, loss, complete parameter gradients,
working and master updates and three-step curves against independent native HF recipes, with TF32 disabled.
DDP BF16 uses autocast and FP32 parameters. ZeRO/Gemini use BF16 working parameters with FP32 masters; FSDP uses
FP32 master parameters and BF16 compute. Export readback replays the original buffer precision, including RoPE.
Neither fused kernels nor their numerical/performance claims are part of the native policy.

## Unified L2 numerical report

The model_zoo entries `transformers_gemma4_e2b_for_causal_lm` and `transformers_gemma4_e4b_for_causal_lm` opt into
`tests.kit.numerics` with native HF references, eager attention, non-reentrant gradient checkpointing and a declared
input/output embedding tie. Both entries use the tiny profiles described above, including PLE and shared KV; these
reports validate reduced configurations, not the downloaded E2B/E4B checkpoints. The generated micro-batches have
unequal valid-token counts. `TokenLoss` uses summed causal cross-entropy and counts shifted, non-padding labels, so
the data-parallel check can test global valid-token normalization.
Padding retains a visible key in every sliding window so the FP64 eager reference stays finite. The unified
independent/resume loss is a random output projection; the original causal CE tests remain separate evidence.

Run from a clean, pushed commit, after allocating the devices and installing this exact source checkout. Create a
new report path for each run. The team submission directory for E2B is:

```text
/mnt/beegfs/ColossalAI/shared-results/numerics/2C/transformers_gemma4_e2b_for_causal_lm/<YYYYMMDD>-<commit>.json
```

```bash
python -m tests.kit.numerics.acceptance \
  --entry transformers_gemma4_e2b_for_causal_lm \
  --out /your/personal/results/<YYYYMMDD>-<commit>.json
```

The default command runs all configurations and writes per-check `passed`, `failed`, `unsupported` and `not_run`
statuses. With one allocated GPU, independent-reference and resume checks can run, while the two-device
`data_parallel` and `data_parallel_tokens` checks must remain `not_run`; this is partial L2 coverage. Read the JSON
even when the command exits 0, since unsupported and unrun checks do not make the exit code fail. Do not loosen
the shared judging rules or overwrite earlier reports. Attach failures to the corresponding issue for review.
See [`tests/kit/numerics/README.md`](../../../tests/kit/numerics/README.md) for supported recipes and report fields.

This integration uses the parameter contract and recipe fixes from PR #6506 at
`bdab6a9e5b4cbdfd649d266f43e3b30891a4a296`, a pending trial dependency when integrated on 2026-11-04. Record the
actual tested commit and dependency status with the report. An E4B tiny report is supplementary to the separate
real-checkpoint single-GPU training run; the team's current 2C report list names E2B, 26B-A4B and DiffusionGemma.

ZeRO optimizer checkpoints preserve FP32 masters in private checkpoint metadata while accepting legacy states.
With `master_weights=True`, Gemini optimizer checkpoints preserve FP32 masters through private
`_colossalai_master_param` metadata in both sharded and unsharded formats. Model exports retain Gemini's existing
precision behavior: working precision on the accumulation path, with gathered reused master chunks retaining FP32.
Save and load both model and optimizer for exact training resume. Legacy optimizer files remain
readable but cannot recover residual precision already lost when they were saved.

## Current validation status

See the task's dated validation record for the tested commit, counts and remaining coverage. Tiny checkpoints do
not establish real E4B checkpoint capacity, convergence or performance. No dependency image upgrade is required.

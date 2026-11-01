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
torchrun --standalone --nproc_per_node=1 examples/language/gemma4/train.py \
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
resume. In particular the main-branch BF16 master-weight resume issue needs separate verification. A first training
step allocates optimizer states; measuring only model load/forward is insufficient for a capacity claim.

## Tests

```bash
# No GPU: native TP=1 policy, PLE/shared-KV gradients, non-reentrant checkpointing,
# FP64/FP32 baseline, three-step update/curve equivalence and exact HF reload.
CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_model_zoo/test_gemma4.py
# Requires two allocated devices. Includes one- and two-device cases, both tiny profiles,
# DDP FP32, ZeRO-1/2 BF16+offload, Gemini BF16, FSDP BF16 and lazy/checkpoint combinations.
python -m pytest -q tests/test_booster/test_plugin/test_gemma4.py
```

The GPU matrix is in batch 5 (excluded from batch 8 to avoid duplication); CPU checks are in both CPU entry lists.
CPU/Gloo checks also compare 1/2-process DDP with single-process gradient accumulation, retaining unequal valid-token
counts, and compare one-step weights. These tests do not claim E2B L1/L2 complete: GPU plugin numerical comparisons,
GPU DP-vs-accumulation equivalence and exact plugin checkpoint resume need their own reports. DDP BF16's existing precision-wrapper problem is not hidden with
an xfail. Neither fused kernels nor their numerical/performance claims are part of the native policy.

## Current validation status

See the task's dated validation record. GPU tests and real E4B training must be run before claiming GPU support;
the initial implementation was developed while all H20 devices were occupied.

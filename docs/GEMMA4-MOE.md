# Gemma 4 26B-A4B: replicated shared experts

The TP=1 Gemma policy replaces only the native HF expert module with
`Gemma4GroupedExperts`. The router, its RMSNorm/scale/FP32 softmax/top-k and learned
per-expert scale, and the decoder's parallel dense and routed branches remain native.
The shared dispatcher runs with `local_only=True`, so a DP process group never becomes
an EP group. Expert parameters are replicated, without EP gradient tags or scaling.

The adapter keeps the original `gate_up_proj` and `down_proj` Parameter objects and
HF checkpoint keys. Optimizers created before adaptation retain both expert tensors.
CPU execution explicitly uses `ReferenceExperts`; CUDA uses `GroupedMMExperts` and
requires `torch._grouped_mm` (PyTorch >=2.8). No CUDA fallback is selected silently.
The development environment is Transformers 5.17.0 / PyTorch 2.13.0+cu130.

```python
from colossalai.shardformer import ShardConfig, ShardFormer

# For a native Gemma4ForCausalLM or Gemma4ForConditionalGeneration:
model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
```

For DDP, ZeRO, Gemini or FSDP without ShardFormer, adapt the model before constructing
the plugin/optimizer:

```python
from colossalai.shardformer.modeling.gemma4 import replace_gemma4_experts

replace_gemma4_experts(model)
```

This path supports replicated experts with TP=1, EP=1 and no PP/SP or optional fused
attention/normalization. It does not add expert-parallel training.

## Numerical validation entry

`transformers_gemma4_26b_a4b_for_causal_lm` is a structural **tiny text model**, not the
full 26B checkpoint: six layers (five sliding, one full), hidden32, dense64/expert32,
eight experts/top2, local/global KV heads2/1 and head dimensions16/32, global K=V,
window4 with sequence8, no PLE or cross-layer KV sharing, GELU and tied embeddings.

Its candidate uses the shared experts; `reference_fn` builds unmodified eager HF.
Both enable non-reentrant gradient checkpointing. The spec records tied parameters,
causal valid-token loss normalization and router choices. No expert parameter is
excluded from gradient checking.

The explicit FP16 initial loss scale is 1024 for both candidate and reference (65536
overflowed in the initial trial). The blank control uses native HF on both sides to
validate the reference recipe; the independent-implementation check still compares
the grouped candidate against native HF with the unchanged judge. BF16 autocast is
applied explicitly around grouped GEMM's inputs/weights because `torch._grouped_mm`
does not perform the `F.linear` autocast conversion itself.

```bash
python -m pytest -q tests/test_model_zoo/test_gemma4_moe.py
# Requires an allocated CUDA device; asserts actual grouped GEMM execution:
python -m pytest -q tests/test_booster/test_plugin/test_gemma4_moe.py
# Requires two allocated CUDA devices for the full DP matrix:
python -m tests.kit.numerics.acceptance \
  --entry transformers_gemma4_26b_a4b_for_causal_lm --out gemma4-moe.json
```

The result file, including failures and not_run/unsupported checks, determines what
was validated. CPU reference checks and fixed-routing layer checks alone do not
establish whole-model CUDA/DP numerical acceptance. Full-size 26B training, long
contexts, multimodal behavior and performance are outside this tiny validation.

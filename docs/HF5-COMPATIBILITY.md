# Transformers 5 compatibility and migration boundaries

Target tested in this checkout: Transformers **5.17.0**, PyTorch **2.11.0**. This table records scopes, not an assertion that the full distributed matrix passes.

## D4 model-family inventory

There are **20 registered policy families**. Excluding the eight core families (BERT, Llama, Qwen2, Qwen3, Mixtral, DeepseekV3, ViT, Whisper) leaves **12**, not 14:

T5, GPT2, GPTJ, OPT, Bloom, SAM, BLIP2, ChatGLM2, Deepseek (legacy/non-V3), Falcon, Mistral, Command/Cohere.

Do not count model heads, remote-code aliases, or Command/Cohere twice.

## Support and gate matrix

| Family | HF5 evidence | Current boundary |
| --- | --- | --- |
| OPT | Removed obsolete attention-class import; tiny eager/SDPA causal-LM forward, gradients and SGD update through actual ShardFormer pass | Native, nonparallel path only; TP/PP/SP, Flash/JIT/fused/fp8/all-optimization are gated pending migration/validation |
| Command/Cohere | StaticCache imports from cache_utils; preserves native HF5 attention instead of unconditional legacy replacement; eager/SDPA causal-LM training checks pass | Same restricted native nonparallel boundary; legacy custom attention/Cache/PP paths are not released |
| BLIP2 | Tiny eager/SDPA multimodal training with valid HF5 image placeholders passes; three obsolete dropout target descriptions removed in HF5 | Native nonparallel path only. Removing targets does **not** establish functional-dropout parallel RNG or TP/Flash/ZBV correctness |
| ChatGLM2 | Remote-code policy module imports, but model code revision/HF5 behavior not validated | HF5 built-in policy gated before lazy import; supply a validated custom policy or use a separately managed legacy environment |
| Deepseek, non-V3 | Legacy remote-code policy, not the native HF5 DeepseekV2 registry | HF5 built-in policy gated; native classes remain unregistered, do not claim V2 support |
| T5/GPT2/GPTJ/Bloom/Falcon/Mistral | Import and tiny descriptor/path audit passes; no full HF5 numerical/distributed acceptance | No new blanket family gate. Existing failures remain visible; old Cache and optional-path contracts require further work |
| SAM | CPU RNG read fixed; TP descriptor next reaches unsupported CPU manual_seed | Remaining platform blocker; not declared an HF5 model incompatibility |
| Eight core families | Four colleague families integrated with HF4/HF5 dispatch; non-EP CPU training checked; HF5 fused expert identity, optimizer ownership/update, optimizer checkpoint and EP gradient regressions executed remotely | The obsolete Mixtral/DeepseekV3 EP identity gate is removed after production four-GPU tests pass. Model-specific restrictions remain; full eight-GPU matrix and physical HF4 regression are not certified |

The three newly validated families require HF5 **>=5.17.0** in this compatibility branch; earlier HF5 minors are not silently treated as tested. Later versions still need regression testing. HF4 branches remain present, but a real HF4 regression was not executed in this D4 pass.

### Validated basic configuration

```python
from colossalai.shardformer import ShardConfig, ShardFormer

config = ShardConfig(enable_tensor_parallelism=False)
model, shared_params = ShardFormer(config).optimize(model)
```

For the new basic scopes, keep optional optimizations and PP/SP disabled. Training probes use FP32, dropout disabled and `use_cache=False`; they do not certify generation/cache, arbitrary dropout, BF16 or long-sequence behavior. BLIP2 HF5 inputs must include the configured image placeholder token repeated for `num_query_tokens`, with valid label masking.

## Error and legacy guidance

Unsupported requests report family, installed Transformers version, requested settings, support boundary and this document path. Automatic policy checks run before model mutation; remote-code gates also run before policy import. Explicit registered built-in policies are checked too. Arbitrary custom policies/subclasses remain the caller's responsibility.

The repository currently declares `transformers==4.51.3`. If an unsupported HF5 path is required, use an **isolated legacy environment**, follow its original dependency declaration and test the desired model/configuration there. This is not a promise that every legacy family passes at that version. Do not downgrade the shared HF5 environment in place.

## Testing and skip accounting

- `tests/test_shardformer/test_hf5_compat/test_d4_compat.py`: actual basic training and negative gate tests; these are executed, not skipped.
- Existing distributed entrypoints for OPT/Command/BLIP2 and unvalidated remote-code ChatGLM2/Deepseek are explicitly skipped under HF5 with the same reason as the gate. Lazy test dependency loading prevents unrelated model-zoo imports before these skip decisions.
- Other families are not blanket skipped. Unexpected import/runtime failures remain failures.
- **SKIP is not PASS**. Four/eight-card coverage, full optimized modes, native generation/Cache and the remaining tail numerical matrix remain pending.

## Integrated loading and pipeline boundaries

- Offline lazy `from_pretrained` records a checkpoint path. Materialization constructs tensors; `Booster.boost` subsequently loads checkpoint weights and clears the path. These are tested separately, including sharded safetensors with tied parameters.
- Strict loading accepts an omitted tied alias only when the same Parameter has a checkpoint-loaded name; missing unrelated weights still fail.
- Padding alias routing applies to the approved **unsharded HybridParallel load** path. It validates all groups before mutation, verifies zero padding and checks logical values. It does not certify sharded alias routing or optimizer checkpoints.
- Lazy loading currently accepts an explicit `torch.dtype`; HF dtype strings/`auto` inference are not certified and remain explicit errors rather than silent guesses.
- Nested scheduler backward and chunk-local P2P metadata are covered by CPU contract tests. Real GPU P2P, full zero-bubble and dynamic-shape behavior require separate validation.

### Legacy inference boundary

HF5 Drafter legacy tuple-cache speculation/resume is not migrated: HF5 removed the legacy Cache conversion APIs. The two legacy-reference tests in `tests/test_infer/test_runtime_compatibility.py` are explicitly skipped under HF5; the nine active PP mask contract cases remain executed. This D1–D5 training migration does not certify legacy inference/cache behavior.

Evidence: `logs/2026-10-12-d4-audit.md`, `logs/2026-10-12-d4-fix/`, `logs/2026-10-13-integration-review/`, and fresh remote validation in `logs/2026-10-20-integration/FINAL.md`.

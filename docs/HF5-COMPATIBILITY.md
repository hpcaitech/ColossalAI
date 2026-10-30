# Transformers 5 compatibility and migration boundaries

Target tested in this checkout: Transformers **5.17.0**, PyTorch **2.11.0**. This table records scopes, not an assertion that the full distributed matrix passes.

## D4 model-family inventory

The original D4 inventory has **20 registered policy families** (the phase-two additions are listed below). Excluding the eight core families (BERT, Llama, Qwen2, Qwen3, Mixtral, DeepseekV3, ViT, Whisper) leaves **12**, not 14:

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


## LLaDA, LLaDA2 and Ouro remote-code adapters (phase two)

These three additional policy families bring the registry to 23 families.
Validated environment: Transformers **5.17.0**, PyTorch **2.13.0+cu130**, CPU,
FP32, tiny random models. The adapters deliberately reject other Transformers
versions pending regression testing.

| Model | Pinned revision | Adapter |
| --- | --- | --- |
| GSAI-ML/LLaDA-8B-Base | 0f2787f2d87eac5eed8a087d5ecd24277e6255b2 | get_llada_model_class |
| inclusionAI/LLaDA2.1-mini | 20e64e2ad21644d0e5248586ed9c942cdd45de0f | get_llada2_model_class |
| inclusionAI/LLaDA2.2-mini | e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5 | get_llada2_model_class |
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | get_ouro_model_class |
| ByteDance/Ouro-2.6B | 1ed04250da1a9936042725d302e81c8fa2ab5abd | get_ouro_model_class |

### What is adapted

- LLaDA: populate HF5 loading metadata without reinitializing the official
  weights. Accept the HF5 tie_weights signature; LLaDA's tied output directly
  uses its input embedding, so no extra state_dict alias is introduced.
- LLaDA2.1/2.2: declare the HF5 target-to-source tied-weight mapping instead
  of the old list. Keep the official MoE, routing and same-position label loss.
- Ouro: construct compatible RotaryEmbedding / Model / LM subclasses;
  provide the standard full-head RoPE initialization protocol; build the HF5
  causal mask before passing it to the official forward's existing mask-map
  interface. The original decoder, repeated shared-layer loop, early-exit gate
  and label loss are inherited. Its tied-weight metadata also uses the HF5 mapping.

Factories do not patch imported modules, shared caches, Transformers registries
or installed files. They cache distinct subclasses per official model class.
They do not download or trust remote code implicitly: callers first load the
official class from an explicitly trusted, pinned snapshot.

### Construction and loading

For example, given an already prepared local Ouro snapshot:

```python
import json
from pathlib import Path

from transformers.dynamic_module_utils import get_class_from_dynamic_module

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.ouro import get_ouro_model_class

snapshot = Path("/path/to/the/pinned/Ouro/snapshot")
raw = json.loads((snapshot / "config.json").read_text())
Config = get_class_from_dynamic_module(raw["auto_map"]["AutoConfig"], str(snapshot), local_files_only=True)
OfficialModel = get_class_from_dynamic_module(
    raw["auto_map"]["AutoModelForCausalLM"], str(snapshot), local_files_only=True
)
config = Config.from_pretrained(snapshot, local_files_only=True)
config.use_cache = False
config._attn_implementation = "eager"
Model = get_ouro_model_class(OfficialModel)

# This loads FULL pretrained weights; use the tiny fixture below for a small CPU check.
model = Model.from_pretrained(
    snapshot, config=config, attn_implementation="eager", local_files_only=True
)
model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
```

Use get_llada_model_class from modeling.llada or get_llada2_model_class
from modeling.llada2_moe for the corresponding official class.
For randomly initialized LLaDA, explicitly call Model(config, init_params=True).
Other families use Model(config). No factory reduces model dimensions implicitly.

Save with model.save_pretrained(output_directory). Reload through the same
adapter class and the saved config, e.g.
Model.from_pretrained(output_directory, config=Config.from_pretrained(output_directory),
attn_implementation="eager", local_files_only=True).
Recreate the factory from the pinned original snapshot in a fresh process.
AutoModel alone still selects the original remote class, not this adapter.
Use save_pretrained/state_dict, not whole-model Python pickling.
Full original-weight loading has not been executed in this validation; tiny
sharded safetensors roundtrips test the construction/loading path and parameter keys.

### Support boundaries and evidence

The policies admit the native path with enable_tensor_parallelism=False and
all optional optimizations disabled. They intentionally perform no kernel
replacement. TP/PP/SP/EP, gradient checkpointing, flash/JIT/fused/fp8 paths and
extra_kwargs are rejected before policy mutation. This is model integration,
not distributed training support.

Use eager attention, no KV cache and no sliding windows. Ouro additionally
requires default full-head RoPE and contiguous positions starting at zero;
its input mask is either absent or a binary 2D padding mask. Runtime cache,
custom position offsets and prebuilt mask mappings are explicitly rejected.
Generation, optimized GPU/BF16 training, long contexts, full pretrained weights,
Booster/ZeRO and the complete diffusion objective remain outside this validation.

tests/kit/remote_models.py provides lazy, offline tiny factories for all five
snapshots without importing the unrelated full model zoo. They are intentionally
not registered in its broad distributed test matrix yet. The two LLaDA2 versions
remain separate fixtures despite sharing a class name.

tests/test_shardformer/test_model/test_shard_remote_models.py checks actual
ShardFormer auto/explicit policy entry, output/loss, every available parameter
gradient and an AdamW update against the official computation with identical
weights. The Ouro test reference uses a test-only constructor bridge plus an
independently built triangular mask; it is not a separately validated HF4
training reference. Tests also cover parameter identity, two loop iterations,
causality, padding, same-process snapshot isolation, sharded save/load, tied
weights, and rejection of unsupported settings. LLaDA uses fixed masked-position
CE only; this is not the full 2D2 diffusion objective.

Prepare the pinned config/code snapshots before testing; no weight download is
needed. Tests skip if snapshots are absent unless COLOSSALAI_REQUIRE_REMOTE_MODELS=1.
The persistent runner requires all five snapshots, records source hashes,
hides GPUs, and preserves the pytest failure status:

```bash
source /home/wzj/venv/bin/activate
bash /workspace/ColossalAI/scripts/phase2/2d1/run.sh adapters
```

Logs are written to /results/2d1/. The runner also executes the existing D4
basic-policy regression suite. Original 2D1 reconnaissance scripts remain
available and continue to expose the unadapted Ouro construction failure.

Validation on 2026-10-28: **105 passed, 0 failed, 0 skipped**. Log: /results/2d1/adapters-20261028-121109-MVBu0Z.log; JUnit: /results/2d1/adapters-validation-20261028-1212.xml.

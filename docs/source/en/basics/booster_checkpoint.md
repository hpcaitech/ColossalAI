# Booster Checkpoint

Author: [Hongxin Liu](https://github.com/ver217)

**Prerequisite:**
- [Booster API](./booster_api.md)

## Introduction

We've introduced the [Booster API](./booster_api.md) in the previous tutorial. In this tutorial, we will introduce how to save and load checkpoints using booster.

## Model Checkpoint

{{ autodoc:colossalai.booster.Booster.save_model }}

Model must be boosted by `colossalai.booster.Booster` before saving. `checkpoint` is the path to saved checkpoint. It can be a file, if `shard=False`. Otherwise, it should be a directory. If `shard=True`, the checkpoint will be saved in a sharded way. This is useful when the checkpoint is too large to be saved in a single file. Our sharded checkpoint format is compatible with [huggingface/transformers](https://github.com/huggingface/transformers), so you can use huggingface `from_pretrained` method to load model from our sharded checkpoint.

{{ autodoc:colossalai.booster.Booster.load_model }}

Model must be boosted by `colossalai.booster.Booster` before loading. It will detect the checkpoint format automatically, and load in corresponding way.

If you want to load a pretrained model from Huggingface while the model is too large to be directly loaded through `from_pretrained` on a single device, a recommended way is to download the pretrained weights to a local directory, and use `booster.load` to load from that directory after boosting the model. Also, the model should be initialized under lazy initialization context to avoid OOM. Here is an example pseudocode:
```python
from colossalai.lazy import LazyInitContext
from huggingface_hub import snapshot_download
...

# Initialize model under lazy init context
init_ctx = LazyInitContext(default_device=get_current_device)
with init_ctx:
     model = LlamaForCausalLM(config)

...

# Wrap the model through Booster.boost
model, optimizer, _, _, _ = booster.boost(model, optimizer)

# download huggingface pretrained model to local directory.
model_dir = snapshot_download(repo_id="lysandre/arxiv-nlp")

# load model using booster.load
booster.load(model, model_dir)
...
```

## Optimizer Checkpoint

{{ autodoc:colossalai.booster.Booster.save_optimizer }}

Optimizer must be boosted by `colossalai.booster.Booster` before saving.

{{ autodoc:colossalai.booster.Booster.load_optimizer }}

Optimizer must be boosted by `colossalai.booster.Booster` before loading.

### Optimizer safetensors format v2

Async optimizer checkpoints use a versioned, data-only safetensors format. Tensor values remain in the tensor data area; parameter groups and nested values use a typed JSON schema. The format preserves lists, tuples, integers, explicit `None`, tensor learning rates and nested tensors. New sharded saves write parameter groups to `optimizer_group.safetensors`; index metadata identifies the group and shard formats. Tensor keys are stable across shards and repeated saves.

The normal loader rejects legacy safetensors files with `non_tensor_keys`, including files produced by older ColossalAI versions. It never falls back to pickle. Tensor-only legacy shards can still be read, but they do not provide the parameter groups required for a complete resume. Existing synchronous PyTorch `.bin` checkpoints retain their format and are read with explicit `weights_only=True`; unsupported Python objects are rejected.

Normal loading of PyTorch optimizer files requires torch 2.10.0 or later because earlier versions have known `weights_only` vulnerabilities. This check applies to full optimizer files, state shards and legacy parameter-group companions; it does not change the project's overall torch dependency or impose this legacy-file check on v2 safetensors. A version check does not certify checkpoint provenance. The explicit trusted migration tool remains available on older torch versions.

To migrate a **trusted** legacy optimizer file or optimizer checkpoint directory, use a new destination:

```bash
colossalai checkpoint migrate-optimizer --trust-legacy-pickle old_optimizer.safetensors new_optimizer.safetensors
colossalai checkpoint migrate-optimizer --trust-legacy-pickle old_optimizer_directory new_optimizer_directory
```

Legacy decoding can execute code. Run this opt-in tool only for trusted inputs in an isolated environment without network access or credentials. It is separate from training loaders, never overwrites an existing destination, and prints input/output SHA-256 hashes as JSON. Conversion is not a way to certify untrusted input. Arbitrary custom Python classes are not supported by v2. Old ColossalAI versions cannot read v2 checkpoints; retain the originals when planning a rollback.

The schema has limits of 16 MiB of JSON, depth 64 and 1,000,000 nodes; integer values are limited to 4096 bits. Unsupported types and oversized schemas raise errors rather than silently dropping optimizer state. Plugin-specific optimizer and sharding constraints still apply.

## LR Scheduler Checkpoint

{{ autodoc:colossalai.booster.Booster.save_lr_scheduler }}

LR scheduler must be boosted by `colossalai.booster.Booster` before saving. `checkpoint` is the local path to checkpoint file.

{{ autodoc:colossalai.booster.Booster.load_lr_scheduler }}

LR scheduler must be boosted by `colossalai.booster.Booster` before loading. `checkpoint` is the local path to checkpoint file.

## Checkpoint design

More details about checkpoint design can be found in our discussion [A Unified Checkpoint System Design](https://github.com/hpcaitech/ColossalAI/discussions/3339).

<!-- doc-test-command: echo  -->

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

## LR Scheduler Checkpoint

{{ autodoc:colossalai.booster.Booster.save_lr_scheduler }}

LR scheduler must be boosted by `colossalai.booster.Booster` before saving. `checkpoint` is the local path to checkpoint file.

{{ autodoc:colossalai.booster.Booster.load_lr_scheduler }}

LR scheduler must be boosted by `colossalai.booster.Booster` before loading. `checkpoint` is the local path to checkpoint file.

## Checkpoint design

More details about checkpoint design can be found in our discussion [A Unified Checkpoint System Design](https://github.com/hpcaitech/ColossalAI/discussions/3339).

<!-- doc-test-command: echo  -->

# Lazy initialization

Author: Hongxin Liu

**Prerequisite**
- [Booster API](../basics/booster_api.md)
- [Booster Plugins](../basics/booster_plugins.md)
- [Booster Checkpoint](../basics/booster_checkpoint.md)

**Related discussion**
- [Lazy initialization of model](https://github.com/hpcaitech/ColossalAI/discussions/3124)

## Introduction

LazyTensor allows DL framework (PyTorch) to execute operations lazily, by storing all operations related to it and reruning them when it's required to be materialized.

LazyInit defers model initialization and it's based on LazyTensor.

This is especially useful when we use model parallelism to train large models, in which case the model cannot fit in GPU memory. Through this, we can initialize model tensors using meta tensor and do static analysis to get shard strategy. And then materialize each tensor and apply the shard strategy. The static analysis can be omitted if the shard strategy is known in advance.

## Usage

You may use lazy initialization when using Gemini, tensor parallelism, pipeline parallelism, and auto-parallelism. In other cases, you may not need to use lazy initialization.

Gemini is compatible with lazy initialization. You can use them together directly.

```python
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from torch.nn import Linear
import colossalai

colossalai.launch_from_torch({})

plugin = GeminiPlugin()
booster = Booster(plugin=plugin)

with LazyInitContext():
    model = Linear(10, 10)

optimizer = HybridAdam(model.parameters())
model, optimizer, *_ = booster.boost(model, optimizer)
```

Note that using lazy initialization when using Gemini is not necessary but recommended. If you don't use lazy initialization, you may get OOM error when initializing the model. If you use lazy initialization, you can avoid this error.

> ⚠ Lazy initialization support for tensor parallelism, pipeline parallelism, and auto-parallelism is still under development.

### Load from pretrained model

We should not load pretrained weight in `LazyInitContext`. If so, lazy initialization is meaningless, as the checkpoint is loaded and it takes much GPU memory. A recommended way is to initialize model from scratch in `LazyInitContext` and load pretrained weight outside `LazyInitContext` after calling `Booster.boost()`.

<!--- doc-test-ignore-start -->
```python
with LazyInitContext():
    model = GPT2LMHeadModel(config)

optimizer = ...
lr_scheduler = ...
dataloader = ...
model, optimizer, lr_scheduler, dataloader = booster.boost(model, optimizer, lr_scheduler, dataloader)

booster.load_model(model, pretrained_path)
```
<!--- doc-test-ignore-end -->

As booster supports both pytorch-fashion checkpoint and huggingface/transformers-fashion pretrained weight, the `pretrained_path` of the above pseudo-code can be either a checkpoint file path or a pretrained weight path. Note that it does not support loading pretrained weights from network. You should download the pretrained weight first and then use a local path.

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 lazy_init.py  -->

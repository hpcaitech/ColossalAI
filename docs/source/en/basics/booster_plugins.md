# Booster Plugins

Author: [Hongxin Liu](https://github.com/ver217)

**Prerequisite:**
- [Booster API](./booster_api.md)

## Introduction

As mentioned in [Booster API](./booster_api.md), we can use booster plugins to customize the parallel training. In this tutorial, we will introduce how to use booster plugins.

We currently provide the following plugins:

- [Low Level Zero Plugin](#low-level-zero-plugin): It wraps the `colossalai.zero.low_level.LowLevelZeroOptimizer` and can be used to train models with zero-dp. It only supports zero stage-1 and stage-2.
- [Gemini Plugin](#gemini-plugin): It wraps the [Gemini](../features/zero_with_chunk.md) which implements Zero-3 with chunk-based and heterogeneous memory management.
- [Torch DDP Plugin](#torch-ddp-plugin): It is a wrapper of `torch.nn.parallel.DistributedDataParallel` and can be used to train models with data parallelism.
- [Torch FSDP Plugin](#torch-fsdp-plugin): It is a wrapper of `torch.distributed.fsdp.FullyShardedDataParallel` and can be used to train models with zero-dp.

More plugins are coming soon.

## Plugins

### Low Level Zero Plugin

This plugin implements Zero-1 and Zero-2 (w/wo CPU offload), using `reduce` and `gather` to synchronize gradients and weights.

Zero-1 can be regarded as a better substitute of Torch DDP, which is more memory efficient and faster. It can be easily used in hybrid parallelism.

Zero-2 does not support local gradient accumulation. Though you can accumulate gradient if you insist, it cannot reduce communication cost. That is to say, it's not a good idea to use Zero-2 with pipeline parallelism.

{{ autodoc:colossalai.booster.plugin.LowLevelZeroPlugin }}

We've tested compatibility on some famous models, following models may not be supported:

- `timm.models.convit_base`
- dlrm and deepfm models in `torchrec`
- `diffusers.VQModel`
- `transformers.AlbertModel`
- `transformers.AlbertForPreTraining`
- `transformers.BertModel`
- `transformers.BertForPreTraining`
- `transformers.GPT2DoubleHeadsModel`

Compatibility problems will be fixed in the future.

> ⚠ This plugin can only load optimizer checkpoint saved by itself with the same number of processes now. This will be fixed in the future.

### Gemini Plugin

This plugin implements Zero-3 with chunk-based and heterogeneous memory management. It can train large models without much loss in speed. It also does not support local gradient accumulation. More details can be found in [Gemini Doc](../features/zero_with_chunk.md).

{{ autodoc:colossalai.booster.plugin.GeminiPlugin }}

### Torch DDP Plugin

More details can be found in [Pytorch Docs](https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel).

{{ autodoc:colossalai.booster.plugin.TorchDDPPlugin }}

### Torch FSDP Plugin

> ⚠ This plugin is not available when torch version is lower than 1.12.0.

> ⚠ This plugin does not support save/load sharded model checkpoint now.

> ⚠ This plugin does not support optimizer that use multi params group.

More details can be found in [Pytorch Docs](https://pytorch.org/docs/main/fsdp.html).

{{ autodoc:colossalai.booster.plugin.TorchFSDPPlugin }}

<!-- doc-test-command: echo  -->

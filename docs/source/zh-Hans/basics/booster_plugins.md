# Booster 插件

作者: [Hongxin Liu](https://github.com/ver217)

**前置教程:**
- [Booster API](./booster_api.md)

## 引言

正如 [Booster API](./booster_api.md) 中提到的，我们可以使用 booster 插件来自定义并行训练。在本教程中，我们将介绍如何使用 booster 插件。

我们现在提供以下插件:

- [Low Level Zero 插件](#low-level-zero-plugin): 它包装了 `colossalai.zero.low_level.LowLevelZeroOptimizer`，可用于使用 Zero-dp 训练模型。它仅支持 Zero 阶段1和阶段2。
- [Gemini 插件](#gemini-plugin): 它包装了 [Gemini](../features/zero_with_chunk.md)，Gemini 实现了基于Chunk内存管理和异构内存管理的 Zero-3。
- [Torch DDP 插件](#torch-ddp-plugin): 它包装了 `torch.nn.parallel.DistributedDataParallel` 并且可用于使用数据并行训练模型。
- [Torch FSDP 插件](#torch-fsdp-plugin): 它包装了 `torch.distributed.fsdp.FullyShardedDataParallel` 并且可用于使用 Zero-dp 训练模型。

更多插件即将推出。

## 插件

### Low Level Zero 插件

该插件实现了 Zero-1 和 Zero-2（使用/不使用 CPU 卸载），使用`reduce`和`gather`来同步梯度和权重。

Zero-1 可以看作是 Torch DDP 更好的替代品，内存效率更高，速度更快。它可以很容易地用于混合并行。

Zero-2 不支持局部梯度累积。如果您坚持使用，虽然可以积累梯度，但不能降低通信成本。也就是说，同时使用流水线并行和 Zero-2 并不是一个好主意。

{{ autodoc:colossalai.booster.plugin.LowLevelZeroPlugin }}

我们已经测试了一些主流模型的兼容性，可能不支持以下模型：

- `timm.models.convit_base`
- dlrm and deepfm models in `torchrec`
- `diffusers.VQModel`
- `transformers.AlbertModel`
- `transformers.AlbertForPreTraining`
- `transformers.BertModel`
- `transformers.BertForPreTraining`
- `transformers.GPT2DoubleHeadsModel`

兼容性问题将在未来修复。

> ⚠ 该插件现在只能加载自己保存的且具有相同进程数的优化器 Checkpoint。这将在未来得到解决。

### Gemini 插件

这个插件实现了基于Chunk内存管理和异构内存管理的 Zero-3。它可以训练大型模型而不会损失太多速度。它也不支持局部梯度累积。更多详细信息，请参阅 [Gemini 文档](../features/zero_with_chunk.md).

{{ autodoc:colossalai.booster.plugin.GeminiPlugin }}

> ⚠ 该插件现在只能加载自己保存的且具有相同进程数的优化器 Checkpoint。这将在未来得到解决。

### Torch DDP 插件

更多详细信息，请参阅 [Pytorch 文档](https://pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel).

{{ autodoc:colossalai.booster.plugin.TorchDDPPlugin }}

### Torch FSDP 插件

> ⚠ 如果 torch 版本低于 1.12.0，此插件将不可用。

> ⚠ 该插件现在还不支持保存/加载分片的模型 checkpoint。

> ⚠ 该插件现在还不支持使用了multi params group的optimizer。

更多详细信息，请参阅 [Pytorch 文档](https://pytorch.org/docs/main/fsdp.html).

{{ autodoc:colossalai.booster.plugin.TorchFSDPPlugin }}

<!-- doc-test-command: echo  -->

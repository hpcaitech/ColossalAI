# Booster Checkpoint

作者: [Hongxin Liu](https://github.com/ver217)

**前置教程:**
- [Booster API](./booster_api.md)

## 引言

我们在之前的教程中介绍了 [Booster API](./booster_api.md)。在本教程中，我们将介绍如何使用 booster 保存和加载 checkpoint。

## 模型 Checkpoint

{{ autodoc:colossalai.booster.Booster.save_model }}

模型在保存前必须被 `colossalai.booster.Booster` 加速。 `checkpoint` 是要保存的 checkpoint 的路径。 如果 `shard=False`，它就是文件。 否则, 它就是文件夹。如果 `shard=True`，checkpoint 将以分片方式保存。当 checkpoint 太大而无法保存在单个文件中时，这很有用。我们的分片 checkpoint 格式与 [huggingface/transformers](https://github.com/huggingface/transformers) 兼容。

{{ autodoc:colossalai.booster.Booster.load_model }}

模型在加载前必须被 `colossalai.booster.Booster` 加速。它会自动检测 checkpoint 格式，并以相应的方式加载。

## 优化器 Checkpoint

> ⚠ 尚不支持以分片方式保存优化器 Checkpoint。

{{ autodoc:colossalai.booster.Booster.save_optimizer }}

优化器在保存前必须被 `colossalai.booster.Booster` 加速。

{{ autodoc:colossalai.booster.Booster.load_optimizer }}

优化器在加载前必须被 `colossalai.booster.Booster` 加速。

## 学习率调度器 Checkpoint

{{ autodoc:colossalai.booster.Booster.save_lr_scheduler }}

学习率调度器在保存前必须被 `colossalai.booster.Booster` 加速。 `checkpoint` 是 checkpoint 文件的本地路径.

{{ autodoc:colossalai.booster.Booster.load_lr_scheduler }}

学习率调度器在加载前必须被 `colossalai.booster.Booster` 加速。 `checkpoint` 是 checkpoint 文件的本地路径.

## Checkpoint 设计

有关 Checkpoint 设计的更多详细信息，请参见我们的讨论 [A Unified Checkpoint System Design](https://github.com/hpcaitech/ColossalAI/discussions/3339).

<!-- doc-test-command: echo  -->

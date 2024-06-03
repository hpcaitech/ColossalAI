# Booster Checkpoint

作者: [Hongxin Liu](https://github.com/ver217)

**前置教程:**
- [Booster API](./booster_api.md)

## 引言

我们在之前的教程中介绍了 [Booster API](./booster_api.md)。在本教程中，我们将介绍如何使用 booster 保存和加载 checkpoint。

## 模型 Checkpoint

{{ autodoc:colossalai.booster.Booster.save_model }}

模型在保存前必须被 `colossalai.booster.Booster` 封装。 `checkpoint` 是要保存的 checkpoint 的路径。 如果 `shard=False`，它就是文件。 否则, 它就是文件夹。如果 `shard=True`，checkpoint 将以分片方式保存，在 checkpoint 太大而无法保存在单个文件中时会很实用。我们的分片 checkpoint 格式与 [huggingface/transformers](https://github.com/huggingface/transformers) 兼容，所以用户可以使用huggingface的`from_pretrained`方法从分片checkpoint加载模型。

{{ autodoc:colossalai.booster.Booster.load_model }}

模型在加载前必须被 `colossalai.booster.Booster` 封装。它会自动检测 checkpoint 格式，并以相应的方式加载。

如果您想从Huggingface加载预训练好的模型，但模型太大以至于无法在单个设备上通过“from_pretrained”直接加载，推荐的方法是将预训练的模型权重下载到本地，并在封装模型后使用`booster.load`直接从本地路径加载。为了避免内存不足，模型需要在`Lazy Initialization`的环境下初始化。以下是示例伪代码：
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

## 优化器 Checkpoint


{{ autodoc:colossalai.booster.Booster.save_optimizer }}

优化器在保存前必须被 `colossalai.booster.Booster` 封装。

{{ autodoc:colossalai.booster.Booster.load_optimizer }}

优化器在加载前必须被 `colossalai.booster.Booster` 封装。

## 学习率调度器 Checkpoint

{{ autodoc:colossalai.booster.Booster.save_lr_scheduler }}

学习率调度器在保存前必须被 `colossalai.booster.Booster` 封装。 `checkpoint` 是 checkpoint 文件的本地路径.

{{ autodoc:colossalai.booster.Booster.load_lr_scheduler }}

学习率调度器在加载前必须被 `colossalai.booster.Booster` 封装。 `checkpoint` 是 checkpoint 文件的本地路径.

## Checkpoint 设计

有关 Checkpoint 设计的更多详细信息，请参见我们的讨论 [A Unified Checkpoint System Design](https://github.com/hpcaitech/ColossalAI/discussions/3339).

<!-- doc-test-command: echo  -->

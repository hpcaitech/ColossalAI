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

### 优化器 safetensors v2 格式

异步优化器 checkpoint 使用带版本的纯数据 safetensors 格式：tensor 保存在数据区，参数组及嵌套结构使用有类型 JSON 描述。它保留列表、元组、整数、显式 None、tensor 学习率和嵌套 tensor。新分片保存将参数组写到 `optimizer_group.safetensors`，索引记录参数组和 shard 的格式；tensor key 在分片与重复保存之间保持稳定。

默认加载器拒绝包含 `non_tensor_keys` 的旧 safetensors 优化器文件，包括旧版 ColossalAI 正常写出的文件，不会回退到 pickle。旧纯 tensor shard 仍可读取，但它本身没有完整恢复需要的参数组。已有同步 PyTorch `.bin` checkpoint 保留原格式，读取显式使用 `weights_only=True`；不支持的 Python 对象会被拒绝。

普通读取 PyTorch 优化器文件要求 torch 2.10.0 或更高版本，因为更早版本存在已知的 `weights_only` 漏洞。这项检查覆盖完整优化器、状态分片和旧参数组伴随文件，不提升整个项目的 torch 最低依赖，也不对 v2 safetensors 应用旧格式门槛。版本检查不能认证文件来源。旧 torch 环境仍可显式迁移可信旧文件。

来源可信的旧优化器文件或目录，可以显式迁移到一个不存在的新目标：

```bash
colossalai checkpoint migrate-optimizer --trust-legacy-pickle old_optimizer.safetensors new_optimizer.safetensors
colossalai checkpoint migrate-optimizer --trust-legacy-pickle old_optimizer_directory new_optimizer_directory
```

旧格式解码仍可能执行代码。此独立工具只用于可信输入，应在无网络、无凭据的隔离环境运行；训练加载器不自动调用它。迁移不覆盖已有目标，完成后输出含输入 / 输出 SHA-256 的 JSON 摘要。迁移不能为不可信输入作安全认证。v2 不支持任意 Python 自定义类；旧版本程序无法读取 v2，回滚计划应保留原文件。

JSON schema 限制为 16 MiB、深度 64、1,000,000 个节点，整数值限制 4096 bit。不支持的类型或超限数据会明确报错，不静默丢弃优化器状态。各插件原有的优化器与分片约束仍然适用。

## 学习率调度器 Checkpoint

{{ autodoc:colossalai.booster.Booster.save_lr_scheduler }}

学习率调度器在保存前必须被 `colossalai.booster.Booster` 封装。 `checkpoint` 是 checkpoint 文件的本地路径.

{{ autodoc:colossalai.booster.Booster.load_lr_scheduler }}

学习率调度器在加载前必须被 `colossalai.booster.Booster` 封装。 `checkpoint` 是 checkpoint 文件的本地路径.

## Checkpoint 设计

有关 Checkpoint 设计的更多详细信息，请参见我们的讨论 [A Unified Checkpoint System Design](https://github.com/hpcaitech/ColossalAI/discussions/3339).

<!-- doc-test-command: echo  -->

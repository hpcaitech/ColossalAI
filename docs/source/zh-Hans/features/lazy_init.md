# 惰性初始化

作者: Hongxin Liu

**前置教程**
- [Booster API](../basics/booster_api.md)
- [Booster 插件](../basics/booster_plugins.md)
- [Booster Checkpoint](../basics/booster_checkpoint.md)

**相关讨论**
- [模型的惰性初始化](https://github.com/hpcaitech/ColossalAI/discussions/3124)

## 引言

LazyTensor 允许深度学习框架 (PyTorch) 延迟执行操作，方法是存储与其相关的所有操作并在需要具体化时重新运行它们。

LazyInit 基于 LazyTensor，并支持延迟模型初始化。

这在我们使用模型并行来训练大型模型时特别有用，在这种情况下模型无法容纳在 GPU 内存中。通过这个，我们可以使用 Meta 张量初始化模型张量并进行静态分析以获得分片策略。然后具体化每个张量并应用分片策略。如果事先知道分片策略，则可以省略静态分析。

## 用法

您可以在使用 Gemini、张量并行、流水线并行和自动并行时使用惰性初始化。在其他情况下，您可能不需要使用惰性初始化。

Gemini 与惰性初始化兼容。您可以直接将它们一起使用。

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

请注意，在使用 Gemini 时使用惰性初始化不是必需的，但建议使用。如果不使用惰性初始化，在初始化模型时可能会出现 OOM 错误。如果使用惰性初始化，则可以避免此错误。

> ⚠ 对张量并行、流水线并行和自动并行的惰性初始化支持仍在开发中。

### 从预训练模型加载

我们不应该在 `LazyInitContext` 中加载预训练权重。如果这样，惰性初始化就没有意义，因为检查点已加载并且需要大量 GPU 内存。推荐的方法是在 `LazyInitContext` 中初始化模型，并在调用 `Booster.boost()` 后在 `LazyInitContext` 之外加载预训练权重。

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

由于 booster 同时支持 pytorch 风格的 checkpoint 和 huggingface/transformers 风格的预训练权重，上述伪代码的 `pretrained_pa​​th` 可以是 checkpoint 文件路径或预训练权重路径。请注意，它不支持从网络加载预训练权重。您应该先下载预训练的权重，然后使用本地路径。

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 lazy_init.py  -->

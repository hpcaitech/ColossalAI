# 懒惰初始化

作者: [Hongxin Liu](https://github.com/ver217)

**前置教程:**
- [Train with booster](../basics/booster_api.md)

## 简介

懒惰初始化延迟了模型的初始化。它能够节省在大模型初始化时的内存占用。

如果你的模型有 `N` 十亿个参数并且你的内存（或显存）为 `M` GB, 我们推荐您在 `4N >= M` 时使用懒惰初始化。否则，懒惰初始化不是必须的。

## 使用

懒惰初始化必须与 booster 一起使用。

### API 参考

{{ autodoc:colossalai.lazy.LazyInitContext }}

### 例子

```python
import colossalai
from colossalai.lazy import LazyInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin

from transformers import LlamaForCausalLM, LlamaConfig, BertForPreTraining

colossalai.launch({})
plugin = GeminiPlugin()
booster = Booster(plugin)

# 1. Initialize model from scratch
# Initialization on cuda will accelerate the initialization process but take more GPU memory.
with LazyInitContext(default_device="cuda"):
    model = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=172, num_hidden_layers=4, num_attention_heads=4))
model, *_ = booster.boost(model)

# 2. Initialize model from pretrained
with LazyInitContext():
    model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")
model, *_ = booster.boost(model)
```

> ⚠️ 使用懒惰初始化加载预训练模型在 colossalai>0.3.3 或主分支上支持。

## 限制

我们提到，懒惰初始化必须与 booster 一起使用。只有几个插件支持它。

| 插件            | 支持情况 | 备注   |
|-----------------|---------|--------|
| Gemini          | 是       |        |
| Hybrid Parallel | 是       |        |
| Low Level Zero  | 否       | 不需要 |
| Torch DDP       | 否       | 不兼容 |
| Torch FSDP      | 否       | 不兼容 |

不是所有的模型都可以懒惰初始化。在某些情况下，一部分参数/缓冲区可能会被提前初始化。但是不用担心，这部分通常只占整个模型的一小部分。

并且一些模型完全不支持，会引发错误。我们测试了 torchvision, diffusers, timm, transformers, torchaudio 和 torchrec 中的模型。以下模型不受支持：

| 模型                          | 分类         |
|-------------------------------|--------------|
| wav2vec2_base                 | torchaudio   |
| hubert_base                   | torchaudio   |
| ViTModel                      | transformers |
| ViTForMaskedImageModeling     | transformers |
| ViTForImageClassification     | transformers |
| Blip2Model                    | transformers |
| Blip2ForConditionalGeneration | transformers |

<!-- doc-test-command: torchrun --standalone --nproc_per_node=2 lazy_init.py  -->

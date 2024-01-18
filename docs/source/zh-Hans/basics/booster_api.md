# Booster API

作者: [Mingyan Jiang](https://github.com/jiangmingyan), [Jianghai Chen](https://github.com/CjhHa1), [Baizhou Zhang](https://github.com/Fridge003)

**预备知识:**

- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)

**示例代码**

<!-- update this url-->

- [使用Booster在CIFAR-10数据集上训练ResNet](https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet)
- [使用Booster在RedPajama数据集上训练Llama-1/2](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama2)

## 简介

在我们的新设计中， `colossalai.booster` 代替 `colossalai.initialize` 将特征(例如，模型、优化器、数据加载器)无缝注入到您的训练组件中。 使用 booster API, 您可以更友好地将我们的并行策略整合到待训练模型中. 调用 `colossalai.booster` 是您进入训练流程前的正常操作。
在下面的章节中，我们将介绍 `colossalai.booster` 是如何工作的以及使用时我们要注意的细节。

### Booster 插件

Booster 插件是管理并行配置的重要组件（eg：gemini 插件封装了 gemini 加速方案）。目前支持的插件如下：

**_HybridParallelPlugin:_** HybridParallelPlugin 插件封装了混合并行的加速解决方案。它提供的接口可以在张量并行，流水线并行以及两种数据并行方法（DDP, Zero）间进行任意的组合。

**_GeminiPlugin:_** GeminiPlugin 插件封装了 gemini 加速解决方案，即基于块内存管理的 ZeRO 优化方案。

**_TorchDDPPlugin:_** TorchDDPPlugin 插件封装了Pytorch的DDP加速方案，实现了模型级别的数据并行，可以跨多机运行。

**_LowLevelZeroPlugin:_** LowLevelZeroPlugin 插件封装了零冗余优化器的 1/2 阶段。阶段 1：切分优化器参数，分发到各并发进程或并发 GPU 上。阶段 2：切分优化器参数及梯度，分发到各并发进程或并发 GPU 上。

**_TorchFSDPPlugin:_** TorchFSDPPlugin封装了 Pytorch的FSDP加速方案，可以用于零冗余优化器数据并行（ZeroDP）的训练。

若想了解更多关于插件的用法细节，请参考[Booster 插件](./booster_plugins.md)章节。

有一些插件支持懒惰初始化，它能节省初始化大模型时的内存占用。详情请参考[懒惰初始化](../features/lazy_init.md)。

### Booster 接口

<!--TODO: update autodoc -->

{{ autodoc:colossalai.booster.Booster }}

## 使用方法及示例

在使用 colossalai 训练时，首先需要在训练脚本的开头启动分布式环境，并创建需要使用的模型、优化器、损失函数、数据加载器等对象。之后，调用`booster.boost` 将特征注入到这些对象中，您就可以使用我们的 booster API 去进行您接下来的训练流程。

以下是一个伪代码示例，将展示如何使用我们的 booster API 进行模型训练:

```python
import torch
from torch.optim import SGD
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

def train():
    # launch colossalai
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')

    # create plugin and objects for training
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = SGD((model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # use booster.boost to wrap the training objects
    model, optimizer, criterion, _, scheduler = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)

    # do training as normal, except that the backward should be called by booster
    x = torch.randn(4, 3, 224, 224)
    x = x.to('cuda')
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # checkpointing using booster api
    save_path = "./model"
    booster.save_model(model, save_path, shard=True, size_per_shard=10, use_safetensors=True)

    new_model = resnet18()
    booster.load_model(new_model, save_path)
```

更多的Booster设计细节请参考这一[页面](https://github.com/hpcaitech/ColossalAI/discussions/3046)

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 booster_api.py  -->

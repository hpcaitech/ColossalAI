# booster 使用

作者: [Mingyan Jiang](https://github.com/jiangmingyan) [Jianghai Chen](https://github.com/CjhHa1)

**预备知识:**

- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)

**示例代码**

<!-- update this url-->

- [使用 booster 训练](https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet/README.md)

## 简介

在我们的新设计中， `colossalai.booster` 代替 `colossalai.initialize` 将特征(例如，模型、优化器、数据加载器)无缝注入您的训练组件中。 使用 booster API, 您可以更友好地将我们的并行策略整合到待训练模型中. 调用 `colossalai.booster` 是您进入训练循环前的基本操作。
在下面的章节中，我们将介绍 `colossalai.booster` 是如何工作的以及使用时我们要注意的细节。

### Booster 插件

Booster 插件是管理并行配置的重要组件（eg：gemini 插件封装了 gemini 加速方案）。目前支持的插件如下：

**_GeminiPlugin:_** GeminiPlugin 插件封装了 gemini 加速解决方案，即基于块内存管理的 ZeRO 优化方案。

**_TorchDDPPlugin:_** TorchDDPPlugin 插件封装了 DDP 加速方案，实现了模型级别的数据并行，可以跨多机运行。

**_LowLevelZeroPlugin:_** LowLevelZeroPlugin 插件封装了零冗余优化器的 1/2 阶段。阶段 1：切分优化器参数，分发到各并发进程或并发 GPU 上。阶段 2：切分优化器参数及梯度，分发到各并发进程或并发 GPU 上。

### Booster 接口

<!--TODO: update autodoc -->

{{ autodoc:colossalai.booster.Booster }}

## 使用方法及示例

在使用 colossalai 训练时，首先需要在训练脚本的开头启动分布式环境，并创建需要使用的模型、优化器、损失函数、数据加载器等对象。之后，调用`colossalai.booster` 将特征注入到这些对象中，您就可以使用我们的 booster API 去进行您接下来的训练流程。

以下是一个伪代码示例，将展示如何使用我们的 booster API 进行模型训练:

```python
import torch
from torch.optim import SGD
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

def train():
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host='localhost')
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = SGD((model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    model, optimizer, criterion, _, scheduler = booster.boost(model, optimizer, criterion, lr_scheduler=scheduler)

    x = torch.randn(4, 3, 224, 224)
    x = x.to('cuda')
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()
    scheduler.step()

    save_path = "./model"
    booster.save_model(model, save_path, True, True, "", 10, use_safetensors=use_safetensors)

    new_model = resnet18()
    booster.load_model(new_model, save_path)
```

[更多的设计细节请参考](https://github.com/hpcaitech/ColossalAI/discussions/3046)

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 booster_api.py  -->

# booster 使用

**预备知识:**
- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)

## 简介
在我们的新设计中， `colossalai.booster` 代替 `colossalai.initialize` 将特征(例如，模型、优化器、数据加载器）无缝注入您的训练组件中。 使用booster API, 您可以更友好的将我们的并行策略整合到模型中. 调用 `colossalai.booster` 是您进入训练循环前的基本操作。
在下面的章节中，我们将介绍 `colossalai.booster` 是如何工作的以及使用中我们要注意的细节。

### Plugin
<p>Plugin是管理并行配置的重要组件（eg：gemini插件封装了gemini加速方案）。目前支持的插件如下：</p>

***GeminiPlugin:*** <p> GeminiPlugin插件封装了 gemini 加速解决方案，即具有基于块的内存管理的 ZeRO优化方案。 </p>

***TorchDDPPlugin:*** <p> TorchDDPPlugin插件封装了DDP加速方案，实现了模块级别的数据并行，可以跨多机运行。 </p>

***LowLevelZeroPlugin:*** <p>LowLevelZeroPlugin插件封装了零冗余优化器的 1/2 阶段。阶段 1：跨数据并行工作器/GPU 的分片优化器状态。阶段 2：分片优化器状态 + 跨数据并行工作者/GPU 的梯度。</p>

### API of booster
Booster.__init__(...):
* 参数:
    * device (str or torch.device): 行训练的设备。默认值：'cuda'。
    * mixed_precision (str or MixedPrecision): 运行训练的混合精度。默认值：None。如果参数是字符串，则它可以是“fp16”、“fp16_apex”、“bf16”或“fp8”。“fp16”将使用 PyTorch AMP，而“fp16_apex”将使用 Nvidia Apex。
    * plugin (Plugin): 运行训练的插件。默认值：None。
    * booster (Booster)


booster.boost(...): 调用此函数来注入特性到对象中。 （例如模型、优化器、标准）
* 参数:
    * model (nn.Module): 被注入的模型对象。
    * optimizer (Optimizer): 被注入的优化器对象。
    * criterion (Callable): 被注入的criterion对象。
    * dataloader (DataLoader): 被注入的dataloader对象.
    * lr_scheduler (LRScheduler): 被注入的lr_scheduler对象.
* 返回值:
    * model, optimizer, criterion, dataloader, lr_scheduler

booster.backward(loss, optimizer): 调用该函数执行反向传播操作。
* 参数:
    * loss (torch.Tensor)
    * optimizer (Optimizer)

booster.no_sync(model) :返回一个上下文管理器，用于禁用跨进程的梯度同步。

booster.save_model(...): 调用此函数以保存模型。
* 参数:
    * model: nn.Module,
    * checkpoint: str,
    * prefix: str = None,
    * shard: bool = False, # if saved as shards
    * size_per_shard: int = 1024  # the max length of shard

booster.load_model(...): 调用该函数加载模型。
* 参数:
    * model: nn.Module,
    * checkpoint: str,
    * strict: bool = True

booster.save_optimizer(...): 调用此函数以保存优化器。
* 参数:
    * optimizer: Optimizer,
    * checkpoint: str,
    * shard: bool = False, # if saved as shards
    * size_per_shard: int = 1024  # the max length of shard

booster.load_optimizer(...): 调用此函数以加载优化器。
* 参数:
    * optimizer: Optimizer,
    * checkpoint: str,

booster.save_lr_scheduler(...): 调用此函数以保存学习率更新器。
* 参数:
    * lr_scheduler: LRScheduler,
    * checkpoint: str,

booster.load_lr_scheduler(...): 调用此函数以加载学习率更新器。
* 参数:
    * lr_scheduler: LRScheduler,
    * checkpoint: str,

## usage

在使用colossalai训练时，首先需要在训练脚本的开头启动分布式环境，并创建需要使用的模型、优化器、损失函数、数据加载器等对象等。之后，调用`colossalai.booster` 将特征注入到这些对象中，您就可以使用我们的booster API去进行您接下来的训练流程。

<P> 以下是一个伪代码示例，将展示如何使用我们的booster API进行模型训练: </p>

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

如果您想运行一个可执行的例子, [请点击](../../../../examples/tutorial/new_api/cifar_resnet/README.md)

[更多的设计细节请参考](https://github.com/hpcaitech/ColossalAI/discussions/3046)

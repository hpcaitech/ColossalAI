# Booster API

Author: [Mingyan Jiang](https://github.com/jiangmingyan) [Jianghai Chen](https://github.com/CjhHa1)

**Prerequisite:**

- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

**Example Code**

- [Train with Booster](https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet/README.md)

## Introduction

In our new design, `colossalai.booster` replaces the role of `colossalai.initialize` to inject features into your training components (e.g. model, optimizer, dataloader) seamlessly. With these new APIs, you can integrate your model with our parallelism features more friendly. Also calling `colossalai.booster` is the standard procedure before you run into your training loops. In the sections below, I will cover how `colossalai.booster` works and what we should take note of.

### Plugin

Plugin is an important component that manages parallel configuration (eg: The gemini plugin encapsulates the gemini acceleration solution). Currently supported plugins are as follows:

**_GeminiPlugin:_** This plugin wraps the Gemini acceleration solution, that ZeRO with chunk-based memory management.

**_TorchDDPPlugin:_** This plugin wraps the DDP acceleration solution, it implements data parallelism at the module level which can run across multiple machines.

**_LowLevelZeroPlugin:_** This plugin wraps the 1/2 stage of Zero Redundancy Optimizer. Stage 1 : Shards optimizer states across data parallel workers/GPUs. Stage 2 : Shards optimizer states + gradients across data parallel workers/GPUs.

### API of booster

{{ autodoc:colossalai.booster.Booster }}

## Usage

In a typical workflow, you should launch distributed environment at the beginning of training script and create objects needed (such as models, optimizers, loss function, data loaders etc.) firstly, then call `colossalai.booster` to inject features into these objects, After that, you can use our booster APIs and these returned objects to continue the rest of your training processes.

A pseudo-code example is like below:

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

[more design details](https://github.com/hpcaitech/ColossalAI/discussions/3046)

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 booster_api.py  -->

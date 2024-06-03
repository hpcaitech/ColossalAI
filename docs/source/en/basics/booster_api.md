# Booster API

Author: [Mingyan Jiang](https://github.com/jiangmingyan), [Jianghai Chen](https://github.com/CjhHa1), [Baizhou Zhang](https://github.com/Fridge003)

**Prerequisite:**

- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

**Example Code**

- [Train ResNet on CIFAR-10 with Booster](https://github.com/hpcaitech/ColossalAI/blob/main/examples/tutorial/new_api/cifar_resnet)
- [Train LLaMA-1/2 on RedPajama with Booster](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/llama2)

## Introduction

In our new design, `colossalai.booster` replaces the role of `colossalai.initialize` to inject features into your training components (e.g. model, optimizer, dataloader) seamlessly. With these new APIs, you can integrate your model with our parallelism features more friendly. Also, calling `colossalai.booster` is the standard procedure before you run into your training loops. In the sections below, we will cover how `colossalai.booster` works and what we should take note of.

### Plugin

Plugin is an important component that manages parallel configuration (eg: The gemini plugin encapsulates the gemini acceleration solution). Currently supported plugins are as follows:

**_HybridParallelPlugin:_** This plugin wraps the hybrid parallel training acceleration solution. It provides an interface for any combination of tensor parallel, pipeline parallel and data parallel strategies including DDP and ZeRO.

**_GeminiPlugin:_** This plugin wraps the Gemini acceleration solution, that ZeRO with chunk-based memory management.

**_TorchDDPPlugin:_** This plugin wraps the DDP acceleration solution of Pytorch. It implements data parallel at the module level which can run across multiple machines.

**_LowLevelZeroPlugin:_** This plugin wraps the 1/2 stage of Zero Redundancy Optimizer. Stage 1 : Shards optimizer states across data parallel workers/GPUs. Stage 2 : Shards optimizer states + gradients across data parallel workers/GPUs.

**_TorchFSDPPlugin:_** This plugin wraps the FSDP acceleration solution of Pytorch and can be used to train models with zero-dp.

More details about usages of each plugin can be found in chapter [Booster Plugins](./booster_plugins.md).

Some plugins support lazy initialization, which can be used to save memory when initializing large models. For more details, please see [Lazy Initialization](../features/lazy_init.md).

### API of booster

{{ autodoc:colossalai.booster.Booster }}

## Usage

In a typical workflow, you should launch distributed environment at the beginning of training script and create objects needed (such as models, optimizers, loss function, data loaders etc.) firstly, then call `booster.boost` to inject features into these objects, After that, you can use our booster APIs and these returned objects to continue the rest of your training processes.

A pseudo-code example is like below:

```python
import torch
from torch.optim import SGD
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

def train():
    # launch colossalai
    colossalai.launch(rank=rank, world_size=world_size, port=port, host='localhost')

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

For more design details please see [this page](https://github.com/hpcaitech/ColossalAI/discussions/3046).

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 booster_api.py  -->

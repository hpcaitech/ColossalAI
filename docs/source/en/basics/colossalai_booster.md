# colossal-ai booster

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

## Introduction
In our new design, `colossalai.booster` replaces the role of `colossalai.initialize` to inject features into your training components (e.g. model, optimizer, dataloader) seamlessly. With these new APIs, user can integrate their model with our parallelism features more friendly. Also calling `colossalai.booster` is the standard procedure before you run into your training loops. In the sections below, I will cover how `colossalai.booster` works and what we should take note of.

### Plugin
<p>Plugin is an important component that manages parallel configuration (eg: The gemini plugin encapsulates the gemini acceleration solution). Currently supported plugins are as follows:</p>

***GeminiPlugin:*** <p> This plugin wrapps the Gemini acceleration solution, that ZeRO with chunk-based memory management. </p>

***TorchDDPPlugin:*** <p>This plugin wrapps the DDP acceleration solution, it implements data parallelism at the module level which can run across multiple machines. </p>

***LowLevelZeroPlugin:*** <p>This plugin wraps the 1/2 stage of Zero Redundancy Optimizer. Stage 1 : Shards optimizer states across data parallel workers/GPUs. Stage 2 : Shards optimizer states + gradients across data parallel workers/GPUs.</p>

### API of booster
Booster.__init__(...):
* Args:
    * device (str or torch.device): The device to run the training. Default: 'cuda'.
    * mixed_precision (str or MixedPrecision): The mixed precision to run the training. Default: None.If the argument is a string, it can be 'fp16', 'fp16_apex', 'bf16', or 'fp8'.'fp16' would use PyTorch AMP while 'fp16_apex' would use Nvidia Apex.
    * plugin (Plugin): The plugin to run the training. Default: None.
* Return:
    * booster (Booster)


booster.boost(...): This function is called to boost objects. (e.g. model, optimizer, criterion).
* Args:
    * model (nn.Module): The model to be boosted.
    * optimizer (Optimizer): The optimizer to be boosted.
    * criterion (Callable): The criterion to be boosted.
    * dataloader (DataLoader): The dataloader to be boosted.
    * lr_scheduler (LRScheduler): The lr_scheduler to be boosted.
* Return:
    * model, optimizer, criterion, dataloader, lr_scheduler

booster.backward(loss, optimizer): This function run the backward operation
* Args:
    * loss (torch.Tensor)
    * optimizer (Optimizer)

booster.no_sync(model) :A context manager to disable gradient synchronizations across processes.

booster.save_model(...): This function is called to save model checkpoints
* Args:
    * model: nn.Module,
    * checkpoint: str,
    * prefix: str = None,
    * shard: bool = False, # if saved as shards
    * size_per_shard: int = 1024  # the max length of shard

booster.load_model(...):
* Args:
    * model: nn.Module,
    * checkpoint: str,
    * strict: bool = True

booster.save_optimizer(...): This function is called to save optimizer checkpoints
* Args:
    * optimizer: Optimizer,
    * checkpoint: str,
    * shard: bool = False, # if saved as shards
    * size_per_shard: int = 1024  # the max length of shard

booster.load_optimizer(...):
* Args:
    * optimizer: Optimizer,
    * checkpoint: str,

booster.save_lr_scheduler(...): This function is called to save lr scheduler checkpoints
* Args:
    * lr_scheduler: LRScheduler,
    * checkpoint: str,

booster.load_lr_scheduler(...):
* Args:
    * lr_scheduler: LRScheduler,
    * checkpoint: str,

## usage
In a typical workflow, you need to launch distributed environment at the beginning of training script and create objects needed (such as models, optimizers, loss function, data loaders etc.) firstly, then call `colossalai.booster` to inject features into these objects, After that, you can use our booster API and these returned objects to continue the rest of your training processes.

<P> A pseudo-code example is like below: </p>

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

if you want to run a example, [click here](../../../../examples/tutorial/new_api/cifar_resnet/README.md)

[more design detailers](https://github.com/hpcaitech/ColossalAI/discussions/3046)

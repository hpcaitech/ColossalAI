# Gradient Accumulation

Author: [Mingyan Jiang](https://github.com/jiangmingyan), [Baizhou Zhang](https://github.com/Fridge003)

**Prerequisite**
- [Training Booster](../basics/booster_api.md)

## Introduction

Gradient accumulation is a common way to enlarge your batch size for training. When training large-scale models, memory can easily become the bottleneck and the batch size can be very small, (e.g. 2), leading to unsatisfactory convergence. Gradient accumulation works by adding up the gradients calculated in multiple iterations, and only update the parameters in the preset iteration.

## Usage

It is simple to use gradient accumulation in Colossal-AI. Just call `booster.no_sync()` which returns a context manager. It accumulate gradients without synchronization, meanwhile you should not update the weights.

## Hands-on Practice

We now demonstrate gradient accumulation. In this example, we let the gradient accumulation size to be 4.

### Step 1. Import libraries in train.py
Create a `train.py` and import the necessary dependencies. The version of `torch` should not be lower than 1.8.1.

```python
import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import DataLoader

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import get_dist_logger
from colossalai.cluster.dist_coordinator import priority_execution
```

### Step 2. Initialize Distributed Environment
We then need to initialize distributed environment. For demo purpose, we uses `launch_from_torch`. You can refer to [Launch Colossal-AI](../basics/launch_colossalai.md) for other initialization methods.

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()
# launch from torch
colossalai.launch_from_torch()
```

### Step 3. Create training components
Build your model, optimizer, loss function, lr scheduler and dataloaders. Note that the root path of the dataset is obtained from the environment variable `DATA`. You may `export DATA=/path/to/data` or change `Path(os.environ['DATA'])` to a path on your machine. Data will be automatically downloaded to the root path.

```python
# define the training hyperparameters
BATCH_SIZE = 128
GRADIENT_ACCUMULATION = 4

# build resnet
model = resnet18(num_classes=10)

# build dataloaders
with priority_execution():
    train_dataset = CIFAR10(root=Path(os.environ.get('DATA', './data')),
                            download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(size=32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                            ]))

# build criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
```

### Step 4. Inject Feature
Create a `TorchDDPPlugin` object to instantiate a `Booster`, and boost these training components.

```python
plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)
train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model, optimizer, criterion, train_dataloader, _ = booster.boost(model=model,
                                                                    optimizer=optimizer,
                                                                    criterion=criterion,
                                                                    dataloader=train_dataloader)
```

### Step 5. Train with Booster
Use booster in a normal training loops, and verify gradient accumulation. `param_by_iter` is to record the distributed training information.
```python
optimizer.zero_grad()
for idx, (img, label) in enumerate(train_dataloader):
        sync_context = booster.no_sync(model)
        img = img.cuda()
        label = label.cuda()
        if idx % (GRADIENT_ACCUMULATION - 1) != 0:
            with sync_context:
                output = model(img)
                train_loss = criterion(output, label)
                train_loss = train_loss / GRADIENT_ACCUMULATION
                booster.backward(train_loss, optimizer)
        else:
            output = model(img)
            train_loss = criterion(output, label)
            train_loss = train_loss / GRADIENT_ACCUMULATION
            booster.backward(train_loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

        ele_1st = next(model.parameters()).flatten()[0]
        param_by_iter.append(str(ele_1st.item()))

        if idx != 0 and idx % (GRADIENT_ACCUMULATION - 1) == 0:
            break

    for iteration, val in enumerate(param_by_iter):
        print(f'iteration {iteration} - value: {val}')

    if param_by_iter[-1] != param_by_iter[0]:
        print('The parameter is only updated in the last iteration')

```


### Step 6. Invoke Training Scripts
To verify gradient accumulation, we can just check the change of parameter values. When gradient accumulation is set, parameters are only updated in the last step. You can run the script using this command:
```shell
colossalai run --nproc_per_node 1 train.py
```

You will see output similar to the text below. This shows gradient is indeed accumulated as the parameter is not updated
in the first 3 steps, but only updated in the last step.

```text
iteration 0, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 1, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 2, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 3, first 10 elements of param: tensor([-0.0141,  0.0464,  0.0507,  0.0321,  0.0356, -0.0150,  0.0172, -0.0118, 0.0222,  0.0473], device='cuda:0', grad_fn=<SliceBackward0>)
```


## Gradient Accumulation on GeminiPlugin

Currently the plugins supporting `no_sync()` method include `TorchDDPPlugin` and `LowLevelZeroPlugin` set to stage 1. `GeminiPlugin` doesn't support `no_sync()` method, but it can enable synchronized gradient accumulation in a torch-like way.

To enable gradient accumulation feature, the argument `enable_gradient_accumulation` should be set to `True` when initializing `GeminiPlugin`. Following is the pseudocode snippet of enabling gradient accumulation for `GeminiPlugin`:
<!--- doc-test-ignore-start -->
```python
...
plugin = GeminiPlugin(..., enable_gradient_accumulation=True)
booster = Booster(plugin=plugin)
...

...
for idx, (input, label) in enumerate(train_dataloader):
    output = gemini_model(input.cuda())
    train_loss = criterion(output, label.cuda())
    train_loss = train_loss / GRADIENT_ACCUMULATION
    booster.backward(train_loss, gemini_optimizer)

    if idx % (GRADIENT_ACCUMULATION - 1) == 0:
        gemini_optimizer.step() # zero_grad is automatically done
...
```
<!--- doc-test-ignore-end -->

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 gradient_accumulation_with_booster.py  -->

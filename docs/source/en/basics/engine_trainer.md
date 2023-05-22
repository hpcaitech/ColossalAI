# Use Engine and Trainer in Training

Author: Shenggui Li, Siqi Mai

> ⚠️ The information on this page is outdated and will be deprecated. Please check [Booster API](../basics/booster_api.md) for more information.

**Prerequisite:**
- [Initialize Features](./initialize_features.md)

## Introduction

In this tutorial, you will learn how to use the engine and trainer provided in Colossal-AI to train your model.
Before we delve into the details, we would like to first explain the concept of engine and trainer.

### Engine

Engine is essentially a wrapper class for model, optimizer and loss function.
When we call `colossalai.initialize`, an engine object will be returned, and it has already been equipped with
functionalities such as gradient clipping, gradient accumulation and zero optimizer as specified in your configuration file.
An engine object will use similar APIs to those of PyTorch training components such that the user has minimum change
to their code.

Below is a table which shows the commonly used APIs for the engine object.

| Component                             | Function                                      | PyTorch                         | Colossal-AI                            |
| ------------------------------------- | --------------------------------------------- | ------------------------------- | -------------------------------------- |
| optimizer                             | Set all gradients to zero before an iteration | optimizer.zero_grad()           | engine.zero_grad()                     |
| optimizer                             | Update the parameters                         | optimizer.step()                | engine.step()                          |
| model                                 | Run a forward pass                            | outputs = model(inputs)         | outputs = engine(inputs)               |
| criterion                             | Calculate the loss value                      | loss = criterion(output, label) | loss = engine.criterion(output, label) |
| criterion                             | Execute back-propagation on the model         | loss.backward()                 | engine.backward(loss)                  |

The reason why we need such an engine class is that we can add more functionalities while hiding the implementations in
the `colossalai.initialize` function.
Imaging we are gonna add a new feature, we can manipulate the model, optimizer, dataloader and loss function in the
`colossalai.initialize` function and only expose an engine object to the user.
The user only needs to modify their code to the minimum extent by adapting the normal PyTorch APIs to the Colossal-AI
engine APIs. In this way, they can enjoy more features for efficient training.

A normal training iteration using engine can be:

```python
import colossalai

# build your model, optimizer, criterion, dataloaders
...

engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                    optimizer,
                                                                    criterion,
                                                                    train_dataloader,
                                                                    test_dataloader)
for img, label in train_dataloader:
    engine.zero_grad()
    output = engine(img)
    loss = engine.criterion(output, label)
    engine.backward(loss)
    engine.step()
```

### Trainer

Trainer is a more high-level wrapper for the user to execute training with fewer lines of code. However, in pursuit of more abstraction, it loses some flexibility compared to engine. The trainer is designed to execute a forward and backward step to perform model weight update. It is easy to create a trainer object by passing the engine object. The trainer has a default value `None` for the argument `schedule`. In most cases, we leave this value to `None` unless we want to use pipeline parallelism. If you wish to explore more about this parameter, you can go to the tutorial on pipeline parallelism.

```python
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks

# build components and initialize with colossalai.initialize
...

# create a logger so that trainer can log on the console
logger = get_dist_logger()

# create a trainer object
trainer = Trainer(
    engine=engine,
    logger=logger
)
```



In trainer, the user can customize some hooks and attach these hooks to the trainer object. A hook object will execute life-cycle methods periodically based on the training scheme. For example,  The `LRSchedulerHook` will execute `lr_scheduler.step()` to update the learning rate of the model during either `after_train_iter` or `after_train_epoch` stages depending on whether the user wants to update the learning rate after each training iteration or only after the entire training epoch. You can store the hook objects in a list and pass it to `trainer.fit` method. `trainer.fit` method will execute training and testing based on your parameters. If `display_process` is True, a progress bar will be displayed on your console to show the training process.

```python
# define the hooks to attach to the trainer
hook_list = [
    hooks.LossHook(),
    hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
    hooks.AccuracyHook(accuracy_func=Accuracy()),
    hooks.LogMetricByEpochHook(logger),
]

# start training
trainer.fit(
    train_dataloader=train_dataloader,
    epochs=NUM_EPOCHS,
    test_dataloader=test_dataloader,
    test_interval=1,
    hooks=hook_list,
    display_progress=True
)
```

If you want to customize your own hook class, you can inherit `hooks.BaseHook` and override the life-cycle methods of your interest. A dummy example to demonstrate how to create a simple log message hook is provided below for your reference.

```python
from colossalai.logging import get_dist_logger
from colossalai.trainer import hooks

class LogMessageHook(hooks.BaseHook):

    def __init__(self, priority=10):
        self._logger = get_dist_logger()

    def before_train(self, trainer):
        self._logger.info('training starts')

    def after_train(self, trainer):
        self._logger.info('training finished')


...

# then in your training script
hook_list.append(LogMessageHook())
```



In the sections below, I will guide you through the steps required to train a ResNet model with both engine and trainer.



## Explain with ResNet

### Overview

In this section we will cover:

1. Use an engine object to train a ResNet34 model on CIFAR10 dataset
2. Use a trainer object to train a ResNet34 model on CIFAR10 dataset

The project structure will be like:

```bash
-- config.py
-- run_resnet_cifar10_with_engine.py
-- run_resnet_cifar10_with_trainer.py
```

Steps 1-4 below are commonly used regardless of using engine or trainer. Thus, steps 1-4 + step 5 will be your `run_resnet_cifar10_with_engine.py` and steps 1-4 + step 6 will form `run_resnet_cifar10_with_trainer.py`.

### Hands-on Practice

#### Step 1. Create a Config File

In your project folder, create a `config.py`. This file is to specify some features you may want to use to train your model. A sample config file is as below:

```python
from colossalai.amp import AMP_TYPE

BATCH_SIZE = 128
NUM_EPOCHS = 200

fp16=dict(
    mode=AMP_TYPE.TORCH
)
```

In this config file, we specify that we want to use batch size 128 per GPU and run for 200 epochs. These two parameters are exposed by `gpc.config`. For example, you can use `gpc.config.BATCH_SIZE` to access the value you store in your config file. The `fp16` configuration tells `colossalai.initialize` to use mixed precision training provided by PyTorch to train the model with better speed and lower memory consumption.

#### Step 2. Initialize Distributed Environment

We need to initialize the distributed training environment. This has been introduced in the tutorial on how to
[launch Colossal-AI](./launch_colossalai.md). For this demonstration, we use `launch_from_torch` and PyTorch launch utility.

```python
import colossalai

# ./config.py refers to the config file we just created in step 1
colossalai.launch_from_torch(config='./config.py')
```

#### Step 3. Create all the training components

In this step, we can create all the components used for training. These components include:

1. Model
2. Optimizer
3. Criterion/loss function
4. Training/Testing dataloaders
5. Learning rate Scheduler
6. Logger



To build these components, you need to import the following modules:

```python
from pathlib import Path
from colossalai.logging import get_dist_logger
import torch
import os
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10
from torchvision.models import resnet34
```



Then build your components in the same way as how to normally build them in your PyTorch scripts. In the script below, we set the root path for CIFAR10 dataset as an environment variable `DATA`. You can change it to any path you like, for example, you can change `root=Path(os.environ['DATA'])` to `root='./data'` so that there is no need to set the environment variable.

```python
# build logger
logger = get_dist_logger()

# build resnet
model = resnet34(num_classes=10)

# build datasets
train_dataset = CIFAR10(
    root='./data',
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.2023, 0.1994, 0.2010]),
        ]
    )
)

test_dataset = CIFAR10(
    root='./data',
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                0.2023, 0.1994, 0.2010]),
        ]
    )
)

# build dataloaders
train_dataloader = get_dataloader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=gpc.config.BATCH_SIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  )

test_dataloader = get_dataloader(dataset=test_dataset,
                                 add_sampler=False,
                                 batch_size=gpc.config.BATCH_SIZE,
                                 num_workers=1,
                                 pin_memory=True,
                                 )

# build criterion
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# lr_scheduler
lr_scheduler = CosineAnnealingLR(optimizer, total_steps=gpc.config.NUM_EPOCHS)
```

#### Step 4. Initialize with Colossal-AI

Next, the essential step is to obtain the engine class by calling `colossalai.initialize`. As stated in `config.py`, we will be using mixed precision training for training ResNet34 model. `colossalai.initialize` will automatically check your config file and assign relevant features to your training components. In this way, our engine object has already been able to train with mixed precision, but you do not have to explicitly take care of it.

```python
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                     optimizer,
                                                                     criterion,
                                                                     train_dataloader,
                                                                     test_dataloader,
                                                                     )
```



#### Step 5. Train with engine

With all the training components ready, we can train ResNet34 just like how to normally deal with PyTorch training.

```python
for epoch in range(gpc.config.NUM_EPOCHS):
    # execute a training iteration
    engine.train()
    for img, label in train_dataloader:
        img = img.cuda()
        label = label.cuda()

        # set gradients to zero
        engine.zero_grad()

        # run forward pass
        output = engine(img)

        # compute loss value and run backward pass
        train_loss = engine.criterion(output, label)
        engine.backward(train_loss)

        # update parameters
        engine.step()

    # update learning rate
    lr_scheduler.step()

    # execute a testing iteration
    engine.eval()
    correct = 0
    total = 0
    for img, label in test_dataloader:
        img = img.cuda()
        label = label.cuda()

        # run prediction without back-propagation
        with torch.no_grad():
            output = engine(img)
            test_loss = engine.criterion(output, label)

        # compute the number of correct prediction
        pred = torch.argmax(output, dim=-1)
        correct += torch.sum(pred == label)
        total += img.size(0)

    logger.info(
        f"Epoch {epoch} - train loss: {train_loss:.5}, test loss: {test_loss:.5}, acc: {correct / total:.5}, lr: {lr_scheduler.get_last_lr()[0]:.5g}", ranks=[0])
```

#### Step 6. Train with trainer

If you wish to train with a trainer object, you can follow the code snippet below:

```python
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks


# create a trainer object
trainer = Trainer(
    engine=engine,
    logger=logger
)

# define the hooks to attach to the trainer
hook_list = [
    hooks.LossHook(),
    hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
    hooks.AccuracyHook(accuracy_func=Accuracy()),
    hooks.LogMetricByEpochHook(logger),
    hooks.LogMemoryByEpochHook(logger)
]

# start training
# run testing every 1 epoch
trainer.fit(
    train_dataloader=train_dataloader,
    epochs=gpc.config.NUM_EPOCHS,
    test_dataloader=test_dataloader,
    test_interval=1,
    hooks=hook_list,
    display_progress=True
)
```



#### Step 7. Start Distributed Training

Lastly, we can invoke the scripts using the distributed launcher provided by PyTorch as we used `launch_from_torch` in Step 2. You need to replace `<num_gpus>` with the number of GPUs available on your machine. This number can be 1 if you only want to use 1 GPU. If you wish to use other launchers, you can refer to the tutorial on How to Launch Colossal-AI.

```bash
# with engine
python -m torch.distributed.launch --nproc_per_node <num_gpus> --master_addr localhost --master_port 29500 run_resnet_cifar10_with_engine.py
# with trainer
python -m torch.distributed.launch --nproc_per_node <num_gpus> --master_addr localhost --master_port 29500 run_resnet_cifar10_with_trainer.py
```

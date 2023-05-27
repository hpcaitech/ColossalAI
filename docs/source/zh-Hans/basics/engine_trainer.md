# 如何在训练中使用 Engine 和 Trainer

作者: Shenggui Li, Siqi Mai

> ⚠️ 此页面上的信息已经过时并将被废弃。请在[Booster API](../basics/booster_api.md)页面查阅更新。

**预备知识:**
- [初始化功能](./initialize_features.md)

## 简介

在本教程中，您将学习如何使用 Colossal-AI 中提供的 Engine 和 Trainer 来训练您的模型。在深入研究细节之前，我们想先解释一下 Engine 和 Trainer 的概念。

### Engine

Engine 本质上是一个模型、优化器和损失函数的封装类。当我们调用 `colossalai.initialize` 时，一个 Engine 对象将被返回，并且配备了在您的配置文件中指定的梯度剪裁、梯度累计和 ZeRO 优化器等功能。

Engine 将使用与 PyTorch 训练组件类似的 API，因此您只需对代码进行微小的修改即可。

下表展示了Engine的常用API。

| 组件                             | 功能                                      | PyTorch                         | Colossal-AI                            |
| ------------------------------------- | --------------------------------------------- | ------------------------------- | -------------------------------------- |
| optimizer                             | 迭代前将所有梯度设置为零 | optimizer.zero_grad()           | engine.zero_grad()                     |
| optimizer                             | 更新参数                         | optimizer.step()                | engine.step()                          |
| model                                 | 进行一次前向计算                            | outputs = model(inputs)         | outputs = engine(inputs)               |
| criterion                             | 计算loss值                      | loss = criterion(output, label) | loss = engine.criterion(output, label) |
| criterion                             | 反向计算         | loss.backward()                 | engine.backward(loss)                  |

我们需要这样一个 Engine 类的原因是，我们可以添加更多的功能，同时将实现隐藏在
`colossalai.initialize` 函数中实现。
假如我们要添加一个新的功能，我们可以在 `colossalai.initialize` 函数中完成对于模型、优化器、数据加载器和损失函数的功能诠释。不管中间的过程有多复杂，最终我们呈现的以及用户需要使用的只有一个 Engine 类，这将十分便捷。
用户只需要在最小范围内修改他们的代码，将普通的 PyTorch APIs 调整为 Colossal-AI
Engine 的 API。通过这种方式，他们可以享受更多的功能来进行有效的训练。

以下是一个简单的例子：

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

Trainer 是一个更高级的封装器，用户可以用更少的代码行来执行训练。 由于 Trainer 的使用会更加简单，相较于 Engine，它会缺少一点灵活性。 Trainer 被设计为进行前向和反向计算来进行模型权重的更新。通过传递 Engine 对象，我们可以很容易地创建一个 Trainer。
Trainer 的参数 `schedule` 默认值是 `None` 。在大多数情况下，除非我们想使用流水线并行，否则我们把这个值设为 `None`。如果您想探索更多关于这个参数的内容，您可以前往流水线并行的相关教程。

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

在 Trainer 中，用户可以定制一些 hooks，并将这些 hooks 附加到 Trainer 上。hook 将根据训练方案定期地执行生命周期函数。例如，基于用户是想在每次训练迭代后还是只在整个训练周期后更新学习率，
`LRSchedulerHook` 将会在 `after_train_iter` 或 `after_train_epoch` 阶段执行 `lr_scheduler.step()` 去为用户更新学习率。您可以将 hook 存储在一个列表中并将其传递给 `trainer.fit` 方法。`trainer.fit` 方法将根据您的参数执行训练和测试。如果 `display_process` 为 True，将在您的控制台显示一个进度条，以显示训练的过程。


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

如果您想定制您的 hook 类，您可以继承 `hooks.BaseHook` 并重写您想要的生命周期方法。下面提供了一个例子来演示如何创建一个简单的关于日志信息的 hook，以供您参考。

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



在下面的章节中，您将会详细地了解到如何用 Engine 和 Trainer 来训练 ResNet 模型。


## ResNet

### 总览

在本节中，我们将介绍：

1. 使用一个 Engine 在 CIFAR10 数据集上训练 ResNet34 模型
2. 使用一个 Trainer 在 CIFAR10 数据集上训练 ResNet34 模型

项目结构如下：

```bash
-- config.py
-- run_resnet_cifar10_with_engine.py
-- run_resnet_cifar10_with_trainer.py
```

对于使用 Engine 或 Trainer，步骤 1-4 是通用的。 因此，步骤 1-4 + 步骤 5 将会是对应 `run_resnet_cifar10_with_engine.py` 而 步骤 1-4 + 步骤6 则对应 `run_resnet_cifar10_with_trainer.py`。

### 牛刀小试

#### 步骤 1. 创建配置文件

在你的项目文件夹中，创建一个 `config.py`。这个文件是用来指定一些您可能想用来训练您的模型的特征。下面是一个配置文件的例子。

```python
from colossalai.amp import AMP_TYPE

BATCH_SIZE = 128
NUM_EPOCHS = 200

fp16=dict(
    mode=AMP_TYPE.TORCH
)
```

在这个配置文件中，我们指定要在每个 GPU 上使用批大小为128，并运行200个 epoch。这两个参数是在 `gpc.config` 中体现的。例如，您可以使用 `gpc.config.BATCH_SIZE` 来访问您存储在配置文件中的批大小值。而 `fp16` 配置则会告诉 `colossalai.initialize` 使用 PyTorch 提供的混合精度训练，以更好的速度和更低的内存消耗来训练模型。

#### 步骤 2. 初始化分布式环境

我们需要初始化分布式训练环境。这在 [启动 Colossal-AI](./launch_colossalai.md) 中有相应的教程。在当前的演示中，我们使用 `launch_from_torch` 和 PyTorch 启用工具。

```python
import colossalai

# ./config.py refers to the config file we just created in step 1
colossalai.launch_from_torch(config='./config.py')
```

#### 步骤 3. 创建所有的训练组件

这时，我们可以创建用于训练的所有组件，包括：

1. 模型
2. 优化器
3. 损失函数
4. 训练/测试数据加载器
5. 学习率调度器
6. 日志记录器



为了构建这些组件，您需要导入以下模块。

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



然后按照通常在PyTorch脚本中构建组件的方式来构建组件。在下面的脚本中，我们将CIFAR10数据集的根路径设置为环境变量 `DATA`。您可以把它改为您想要的任何路径，例如，您可以把 `root=Path(os.environ['DATA'])` 改为 `root='./data'` ，这样就不需要设置环境变量。

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

#### 步骤 4. 用 Colossal-AI 进行初始化

接下来，重要的一步是通过调用 `colossalai.initialize` 获得 Engine。正如 `config.py` 中所述，我们将使用混合精度训练来训练 ResNet34 模型。`colossalai.initialize` 将自动检查您的配置文件，并将相关特征分配给您的训练组件。这样一来，我们的 Engine 已经能够进行混合精度训练，而您不需要进行额外的处理。

```python
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                     optimizer,
                                                                     criterion,
                                                                     train_dataloader,
                                                                     test_dataloader,
                                                                     )
```



#### 步骤 5. 用 Engine 进行训练

当所有的训练组件都准备好后，我们就可以像使用 PyTorch 一样训练 ResNet34 了。

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

#### 步骤 6. 用 Trainer 进行训练

如果您想用 Trainer 进行训练，您可以参考下面的代码进行您的实验。


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



#### 步骤 7. 开始分布式训练

最后，我们可以使用 PyTorch 提供的分布式启动器来调用脚本，因为我们在步骤2中使用了 `launch_from_torch`。您需要把`<num_gpus>` 替换成您机器上可用的GPU数量。如果您只想使用一个 GPU，您可以把这个数字设为1。如果您想使用其他的启动器，请您参考如何启动 Colossal-AI 的教程。


```bash
# with engine
python -m torch.distributed.launch --nproc_per_node <num_gpus> --master_addr localhost --master_port 29500 run_resnet_cifar10_with_engine.py
# with trainer
python -m torch.distributed.launch --nproc_per_node <num_gpus> --master_addr localhost --master_port 29500 run_resnet_cifar10_with_trainer.py
```

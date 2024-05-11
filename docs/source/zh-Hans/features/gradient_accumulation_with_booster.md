# 梯度累积

作者: [Mingyan Jiang](https://github.com/jiangmingyan), [Baizhou Zhang](https://github.com/Fridge003)

**前置教程**
- [训练中使用Booster](../basics/booster_api.md)

## 引言

梯度累积是一种常见的增大训练 batch size 的方式。 在训练大模型时，内存经常会成为瓶颈，并且 batch size 通常会很小（如2），这导致收敛性无法保证。梯度累积将多次迭代的梯度累加，并仅在达到预设迭代次数时更新参数。

## 使用

在 Colossal-AI 中使用梯度累积非常简单，booster提供no_sync返回一个上下文管理器，在该上下文管理器下取消同步并且累积梯度。

## 实例

我们将介绍如何使用梯度累积。在这个例子中，梯度累积次数被设置为4。

### 步骤 1. 在 train.py 导入相关库
创建train.py并导入必要依赖。 `torch` 的版本应不低于1.8.1。

```python
import os
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import get_dist_logger
from colossalai.cluster.dist_coordinator import priority_execution
```

### 步骤 2. 初始化分布式环境

我们需要初始化分布式环境。为了快速演示，我们使用`launch_from_torch`。你可以参考 [Launch Colossal-AI](../basics/launch_colossalai.md)使用其他初始化方法。

```python
# initialize distributed setting
parser = colossalai.get_default_parser()
args = parser.parse_args()

# launch from torch
colossalai.launch_from_torch()

```

### 步骤 3. 创建训练组件

构建你的模型、优化器、损失函数、学习率调整器和数据加载器。注意数据集的路径从环境变量`DATA`获得。你可以通过 `export DATA=/path/to/data` 或 `Path(os.environ['DATA'])`，在你的机器上设置路径。数据将会被自动下载到该路径。

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

### 步骤 4. 注入特性
创建一个`TorchDDPPlugin`对象，并作为参实例化`Booster`， 调用`booster.boost`注入特性。

```python
plugin = TorchDDPPlugin()
booster = Booster(plugin=plugin)
train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model, optimizer, criterion, train_dataloader, _ = booster.boost(model=model,
                                                                    optimizer=optimizer,
                                                                    criterion=criterion,
                                                                    dataloader=train_dataloader)
```


### 步骤 5. 使用booster训练
使用booster构建一个普通的训练循环，验证梯度累积。 `param_by_iter` 记录分布训练的信息。
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

### 步骤 6. 启动训练脚本
为了验证梯度累积，我们可以只检查参数值的变化。当设置梯度累加时，仅在最后一步更新参数。您可以使用以下命令运行脚本：
```shell
colossalai run --nproc_per_node 1 train.py
```

你将会看到类似下方的文本输出。这展现了梯度虽然在前3个迭代中被计算，但直到最后一次迭代，参数才被更新。

```text
iteration 0, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 1, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 2, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 3, first 10 elements of param: tensor([-0.0141,  0.0464,  0.0507,  0.0321,  0.0356, -0.0150,  0.0172, -0.0118, 0.0222,  0.0473], device='cuda:0', grad_fn=<SliceBackward0>)
```

## 在Gemini插件中使用梯度累积

目前支持`no_sync()`方法的插件包括 `TorchDDPPlugin` 和 `LowLevelZeroPlugin`（需要设置参数`stage`为1）. `GeminiPlugin` 不支持 `no_sync()` 方法, 但是它可以通过和`pytorch`类似的方式来使用同步的梯度累积。

为了开启梯度累积功能，在初始化`GeminiPlugin`的时候需要将参数`enable_gradient_accumulation`设置为`True`。以下是 `GeminiPlugin` 进行梯度累积的伪代码片段:
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

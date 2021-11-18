# 引擎与训练器

## 引擎

为了更好的理解我们的`Engine`类是如何工作的，我们首先需要了解常见引擎中进程函数的概念。进程函数控制数据集中一个批次的行为，`Engine`类控制的正是该进程函数。我们在下方的代码块中给出了一个标准的进程函数例子。

```python
def process_function(dataloader, model, criterion, optim):
    optim.zero_grad()
    data, label = next(dataloader)
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optim.setp()
```

在`ignite.engine`与`keras.engine`中，进程函数需要由用户提供，然而，用户很难为流水线并行编写进程函数。为了向用户提供方便的混合并行，我们提供了具备强大功能的`Engine`
类，该类支持流水线并行，并提供前向传播后向传播不交织的策略。同时，您可以在`Engine`类中使用您事先定义好的学习率调度器来在训练过程中调整学习率。

您在构造引擎时只需要定义`model`、`criterion`、`optimizer`、`lr_scheduler`与`schedule`等变量即可，下面的代码块给出了一个这样的例子。
**如果你使用`colossalai.initialize`的话，engine会从config文件里自动构建。**

```python
import torch
import torch.nn as nn
import torchvision.models as models
import colossalai
from colossalai.engine import Engine

model = models.resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model)
lr_scheduler = colossalai.nn.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
schedule = colossalai.engine.NoPipelineSchedule()

MyEngine = Engine(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    step_schedule=schedule
)
```

更多该类的相关信息可以在API信息中找到。

## 训练器

要了解如何个性化适应您需求的训练器，首先需要了解我们的`Trainer`类。

`Trainer`类旨在让科研工作者和工程师更加方便地使用我们的系统，您不需要自己写脚本，只需要调用`Trainer`类来构造您的训练器即可，就像下面的代码块中所做的。

```python
MyTrainer = Trainer(my_trainer)
```

在此之后，您可以使用`fit`方法来训练或调用您的模型。除此之外，为了让我们的`Trainer`
类拥有更强大的功能，我们加入了一系列方便您使用的工具。例如，您可以在训练过程中持续监测并记录模型目前的运行状态和表现，这些功能都是通过钩子函数来实现的。我们提供的`BasicHook`
类让您可以在指定时间执行您的钩子函数。如下方的代码块所示，我们事先为您定义好了一些实用的钩子函数，您需要做的就是找到符合您需求的钩子函数。更多该类的相关信息可以在API信息中找到。

```python
hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogTimingByEpochHook'),
    dict(type='LogMemoryByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    dict(type='TensorboardHook', log_dir='./tfb_logs'),
    dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]
```

上面这些钩子函数可以记录模型性能指标，训练时间，显存使用等信息，并在每一个epoch结束后将这些信息写入到日志中。除此之外，这些钩子函数还可以即时输出当前的损失以及准确率，让用户可以监测模型的性能。

### 钩子函数

如果您有个性化需求，您可以继承我们的`BaseHook`类并添加您的钩子函数，或者继承我们的`MetricHook`来编写您需要的度量标准。这些钩子函数可以在`Trainer`
生命周期的12个时间点被执行。更多该类的相关信息可以在API信息中找到。

### 度量标准

您可以通过继承我们的`Metric`类来提供您需要的度量标准，该类需要与`MetricHook`类一同使用。当您编写您的度量标准钩子函数时，请用心设置您的优先级来确保该钩子函数的优先级高于那些需要度量结果的钩子函数。

我们已经为您定义好了一些度量标准钩子函数在`runner.states['metrics']`供您参考。

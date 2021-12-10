# Colossal-AI Engine & Customize Your Trainer

## Colossal-AI engine

To better understand how `Engine` class works, let's start from the conception of the process function in common
engines. The process function usually controls the behavior over a batch of a dataset, `Engine` class just controls the
process function. Here we give a standard process function in the following code block.

```python
def process_function(dataloader, model, criterion, optim):
    optim.zero_grad()
    data, label = next(dataloader)
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optim.setp()
```

The engine class is a high-level wrapper of these frequently-used functions while preserving the PyTorch-like function signature and integrating with our features.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import colossalai
from colossalai.engine import Engine
from torchvision.datasets import CIFAR10

model = models.resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

dataset = CIFAR10(...)
dataloader = colossalai.utils.get_dataloader(dataset)

engine, dataloader, _,  _ = colossalai.initialize(model, optimizer, criterion, dataloader)

# exmaple of a training iteratio
for img, label in dataloader:
    engine.zero_grad()
    output = engine(img)
    loss = engine.criterion(output, label)
    engine.backward(loss)
    engine.step()

```

More information regarding the class can be found in the API references.

## Customize your trainer

### Overview

To learn how to customize a trainer which meets your needs, let's first give a look at the `Trainer` class. We highly
recommend that you read *Get Started*
section and *Colossal-AI engine* first.

The `Trainer` class enables researchers and engineers to use our system more conveniently. Instead of having to write
your own scripts, you can simply construct your own trainer by calling the `Trainer` class, just like what we did in the
following code block.

```python
trainer = Trainer(engine)
```

After that, you can use the `fit` method to train or evaluate your model. In order to make our `Trainer` class even more
powerful, we incorporate a set of handy tools to the class. For example, you can monitor or record the running states
and metrics which indicate the current performance of the model. These functions are realized by hooks. The `BasicHook`
class allows you to execute your hook functions at specified time. We have already created some practical hooks for you,
as listed below. What you need to do is just picking the right ones which suit your needs. Detailed descriptions of the
class can be found in the API references.

These hook functions will record metrics, elapsed time and memory usage and write them to log after each epoch. Besides,
they print the current loss and accuracy to let users monitor the performance of the model.

```python
import colossalai
from colossalai.trainer import hooks, Trainer
from colossalai.utils import MultiTimer
from colossalai.logging import get_dist_logger

... = colossalai.initialize(...)

timer = MultiTimer()
logger = get_dist_logger()

# if you want to save log to file
logger.log_to_file('./logs/')

trainer = Trainer(
    engine=engine,
    timer=timer,
    logger=logger
)

hook_list = [
    hooks.LossHook(),
    hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
    hooks.AccuracyHook(),
    hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
    hooks.LogMetricByEpochHook(logger),
    hooks.LogMemoryByEpochHook(logger),
    hooks.LogTimingByEpochHook(timer, logger),
    hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
]

trainer.fit(
    train_dataloader=train_dataloader,
    epochs=NUM_EPOCHS,
    test_dataloader=test_dataloader,
    test_interval=1,
    hooks=hook_list,
    display_progress=True
)

```

### Hook

If you have your specific needs, feel free to extend our `BaseHook` class to add your own functions, or our `MetricHook`
class to write a metric collector. These hook functions can be called at different stage in the trainer's life cycle.
Besides, you can define the priorities of all hooks to arrange the execution order of them. More information can be
found in the API references.

### Metric

You can write your own metrics by extending our `Metric` class. It should be used with the `MetricHook` class. When your
write your own metric hooks, please set the priority carefully and make sure the hook is called before other hooks which
might require the results of the metric hook.

We've already provided some metric hooks and we store metric objects in `runner.states['metrics']`. It is a dictionary
and metrics can be accessed by their names.

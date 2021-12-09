# Build your engine & Customize your trainer

## Build your engine

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

In `ignite.engine` or `keras.engine`, the process function is always provided by users. However, it is tricky for users
to write their own process functions for pipeline parallelism. Aiming at offering accessible hybrid parallelism for
users, we provide the powerful `Engine` class. This class enables pipeline parallelism and offers
one-forward-one-backward non-interleaving strategy. Also, you can use pre-defined learning rate scheduler in
the `Engine` class to adjust learning rate during training.

In order to build your engine, just set variables `model`, `criterion`, `optimizer`, `lr_scheduler` and `schedule`. The
following code block provides an example. **The engine is automatically created from the config file for you if you
start with `colossalai.initialize`.**

```python
import torch
import torch.nn as nn
import torchvision.models as models
import colossalai
from colossalai.engine import Engine

model = models.resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
schedule = colossalai.engine.NonPipelineSchedule()

MyEngine = Engine(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    step_schedule=schedule
)
```

More information regarding the class can be found in the API references.

## Customize your trainer

### Overview

To learn how to customize a trainer which meets your needs, let's first give a look at the `Trainer` class. We highly
recommend that you read *Get Started*
section and *Build your engine* first.

The `Trainer` class enables researchers and engineers to use our system more conveniently. Instead of having to write
your own scripts, you can simply construct your own trainer by calling the `Trainer` class, just like what we did in the
following code block.

```python
MyTrainer = Trainer(my_engine)
```

After that, you can use the `fit` method to train or evaluate your model. In order to make our `Trainer` class even more
powerful, we incorporate a set of handy tools to the class. For example, you can monitor or record the running states
and metrics which indicate the current performance of the model. These functions are realized by hooks. The `BasicHook`
class allows you to execute your hook functions at specified time. We have already created some practical hooks for you,
as listed below. What you need to do is just picking the right ones which suit your needs. Detailed descriptions of the
class can be found in the API references.

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

These hook functions will record metrics, elapsed time and memory usage and write them to log after each epoch. Besides,
they print the current loss and accuracy to let users monitor the performance of the model.

### Hook

If you have your specific needs, feel free to extend our `BaseHook` class to add your own functions, or our `MetricHook`
class to write a metric collector. These hook functions can be called at twelve timing in the trainer's life cycle.
Besides, you can define the priorities of all hooks to arrange the execution order of them. More information can be
found in the API references.

### Metric

You can write your own metrics by extending our `Metric` class. It should be used with the `MetricHook` class. When your
write your own metric hooks, please set the priority carefully and make sure the hook is called before other hooks which
might require the results of the metric hook.

We've already provided some metric hooks and we store metric objects in `runner.states['metrics']`. It is a dictionary
and metrics can be accessed by their names.

# Build your engine & Customize your trainer

## Build your engine

To better understand the function of `Engine` class, you should know the conception of the process function in common engines. The process function usually controls the behavior over a batch of a dataset, `Engine` class just controls the process function. For example, common process function looks like this:

```python
def process_function(dataloader, model, criterion, optim):
    optim.zero_grad()
    data, label = next(dataloader)
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optim.setp()
```

In `ignite.engine` or `keras.engine`, the process function is always provided by users. However, it is hard for users to write their own functions for pipeline parallelism.  Aiming at accessible hybrid parallelism for users, we provide powerful `Engine` class. It enables pipeline parallelism and offers 1F1B non-interleaving strategy. Also, you can use pre-defined learning rate scheduler in your `Engine` to adjust learning rate during training.

In order to build your engine, just set model, criterion, optimizer, learning rate scheduler and schedule. Consider the following code as an example.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import colossalai


model = models.resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model)
lr_scheduler = colossalai.nn.lr_scheduler.CosineAnnealingLR(optimizer, 1000)
schedule = colossalai.engine.schedule.NoPipelineSchedule()

MyEngine = Engine(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    schedule=schedule
)
```

More information is in API reference.



## Customize your trainer

### Overview

Before starting to learn how to customize a trainer meeting your need, you should have a basic understanding about the function of `Trainer`. We recommend you to read *Get Started* section and *Build your engine* first. 

Trainer class tends to enable researchers and engineers to use our framework more conveniently, instead of writing their own scripts, we provide `Trainer` class and you can simply construct it with your own `Engine` by calling `MyTrainer = Trainer(MyEngine)`.  Then use method `fit` to train or evaluate your model. In order to make our `Trainer` class more powerful, we add some useful features to it, such as monitor or record running states and metrics which indicate model's performance, or save after a training epoch. 

To accomplish  that, specific actions must be added to the training or evaluation. `BaseHook` class allow you to add desired actions in specific time points. We have already created practical hooks for those useful features. What you need to do is just picking the hooks you want. 

More detailed class descriptions can be found in API reference.

### Example

```python
hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogTimingByEpochHook'),
    dict(type='LogMemoryByEpochHook'),
    dict(type='AccuracyHook'),
    dict(type='LossHook'),
    # dict(type='TensorboardHook', log_dir='./tfb_logs'),
    # dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    # dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]
```

Above hooks will record metrics, used time and memory usage to log every epoch. Also it prints loss and accuracy to let users monitor the performance of the model.

### Hook

You can extend our `BaseHook` class. Hooks can be called at twelve time points. More detailed information can be found in API reference.

Or extend from `MetricHook` to write a metric collector. You should also use the decorator `@HOOKS.register_module` for your own hook class, and import it in your main python script.

For `after_train_iter()`, it receives the output of engine per iteration, which is a list including output, label and loss.

Note that you can define the priority to arrange the execution order of all hooks.

### Metric

You can write your own metric by extending `Metric` class.  It is always used with `MetricHook`. If you write your own metric hooks, please set the priority carefully and make sure is called before other hooks which may use the results of metrics.

We've already provided some metric hooks. We store metric objects in `runner.states['metrics']`. It is a dictionary and you can use the name of the metric to access it.
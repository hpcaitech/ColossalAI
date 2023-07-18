# Initialize Features

Author: Shenggui Li, Siqi Mai

> ⚠️ The information on this page is outdated and will be deprecated. Please check [Booster API](../basics/booster_api.md) for more information.

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

## Introduction

In this tutorial, we will cover the use of `colossalai.initialize` which injects features into your training components
(e.g. model, optimizer, dataloader) seamlessly. Calling `colossalai.initialize` is the standard procedure before you run
into your training loops.

In the section below, I will cover how `colossalai.initialize` works and what we should take note  of.

## Usage

In a typical workflow, we will launch distributed environment at the beginning of our training script.
Afterwards, we will instantiate our objects such as model, optimizer, loss function, dataloader etc. At this moment, `colossalai.initialize`
can come in to inject features into these objects. A pseudo-code example is like below:

```python
import colossalai
import torch
...


# launch distributed environment
colossalai.launch(config='./config.py', ...)

# create your objects
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_dataloader = MyTrainDataloader()
test_dataloader = MyTrainDataloader()

# initialize features
engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                     optimizer,
                                                                     criterion,
                                                                     train_dataloader,
                                                                     test_dataloader)
```

The `colossalai.initialize` function will return an `Engine` object. The engine object is a wrapper
for model, optimizer and loss function. **The engine object will run with features specified in the config file.**
More details about the engine can be found in the [Use Engine and Trainer in Training](./engine_trainer.md).

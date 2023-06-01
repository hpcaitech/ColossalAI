# Model Checkpoint

Author : Guangyang Lu

> ⚠️ The information on this page is outdated and will be deprecated. Please check [Booster Checkpoint](../basics/booster_checkpoint.md) for more information.

**Prerequisite:**
- [Launch Colossal-AI](./launch_colossalai.md)
- [Initialize Colossal-AI](./initialize_features.md)

**Example Code:**
- [ColossalAI-Examples Model Checkpoint](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/utils/checkpoint)

**This function is experiential.**

## Introduction

In this tutorial, you will learn how to save and load model checkpoints.

To leverage the power of parallel strategies in Colossal-AI, modifications to models and tensors are needed, for which you cannot directly use `torch.save` or `torch.load`  to save or load model checkpoints. Therefore, we have provided you with the API to achieve the same thing.

Moreover, when loading, you are not demanded to use the same parallel strategy as saving.

## How to use

### Save

There are two ways to train a model in Colossal-AI, by engine or by trainer.
**Be aware that we only save the `state_dict`.** Therefore, when loading the checkpoints, you need to define the model first.

#### Save when using engine

```python
from colossalai.utils import save_checkpoint
model = ...
engine, _, _, _ = colossalai.initialize(model=model, ...)
for epoch in range(num_epochs):
    ... # do some training
    save_checkpoint('xxx.pt', epoch, model)
```

#### Save when using trainer
```python
from colossalai.trainer import Trainer, hooks
model = ...
engine, _, _, _ = colossalai.initialize(model=model, ...)
trainer = Trainer(engine, ...)
hook_list = [
            hooks.SaveCheckpointHook(1, 'xxx.pt', model)
            ...]

trainer.fit(...
            hook=hook_list)
```

### Load

```python
from colossalai.utils import load_checkpoint
model = ...
load_checkpoint('xxx.pt', model)
... # train or test
```

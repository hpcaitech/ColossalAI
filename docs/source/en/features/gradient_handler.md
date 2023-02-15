# Gradient Handler

Author: Shenggui Li, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Use Engine and Trainer in Training](../basics/engine_trainer.md)

**Example Code**
- [ColossalAI-Examples Gradient Handler](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_handler)

## Introduction

In distributed training, gradient synchronization is required at the end of each iteration. This is important because we
need to make sure the parameters are updated with the same gradients in different machines so that the resulting parameters
are the same. This is often seen in data parallel as the model is replicated across data parallel ranks.

In Colossal-AI, we provide an interface for users to customize how they want to handle the synchronization. This brings
flexibility in cases such as implementing a new parallelism method.

When gradient handlers are used, PyTorch `DistributedDataParallel` will not be used as it will synchronize automatically.

## Customize Your Gradient Handlers

To implement a customized gradient handler, you need to follow these steps.
1. inherit `BaseGradientHandler` in Colossal-AI.
2. register the gradient handler into the `GRADIENT_HANDLER`.
3. implement `handle_gradient` method.

```python
from colossalai.registry import GRADIENT_HANDLER
from colossalai.engine.gradient_handler import BaseGradientHandler


@GRADIENT_HANDLER.register_module
class MyGradientHandler(BaseGradientHandler):

    def handle_gradient(self):
        do_something()


```


## Usage

To use a gradient handler, you need to specify your gradient handler in the config file. The gradient handler
will be automatically built and attached to the engine.

```python
gradient_handler = [dict(type='MyGradientHandler')]
```


### Hands-On Practice

We provide a [runnable example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_handler)
to demonstrate the use of gradient handler. In this example, we used `DataParallelGradientHandler` instead of PyTorch
`DistributedDataParallel` for data parallel training.

```shell
python -m torch.distributed.launch --nproc_per_node 4 --master_addr localhost --master_port 29500  train_with_engine.py
```

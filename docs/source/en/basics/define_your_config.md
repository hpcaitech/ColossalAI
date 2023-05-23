# Define Your Configuration

Author: Guangyang Lu, Shenggui Li, Siqi Mai

> ⚠️ The information on this page is outdated and will be deprecated. Please check [Booster API](../basics/booster_api.md) for more information.


**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)


## Introduction

In Colossal-AI, a configuration file is required to specify the features the system will inject into the training process.
In this tutorial, we will introduce you how to construct your configuration file and how this config file will be used.
Using configuration file has several advantages:

1. You can store your feature configuration and training hyper-parameters in different configuration files
2. New features released in the future can be specified in the configuration without code change in the training script

In this tutorial, we will cover how to define your configuration file.

## Configuration Definition

In a configuration file, there are two types of variables. One serves as feature specification and the other serves
as hyper-parameters. All feature-related variables are reserved keywords. For example, if you want to use mixed precision
training, you need to use the variable name `fp16` in the config file and follow a pre-defined format.

### Feature Specification

There is an array of features Colossal-AI provides to speed up training. Each feature is defined by a corresponding field
in the config file. In this tutorial, we are not giving the config details for all the features, but rather we are providing
an illustration of how to specify a feature. **The details of each feature can be found in its respective tutorial.**

To illustrate the use of config file, we use mixed precision training as an example here. In order to do so, you need to
follow the steps below.

1. create a configuration file (e.g. `config.py`, the file name can be anything)
2. define the mixed precision configuration in the config file. For example, in order to use mixed precision training
natively provided by PyTorch, you can just write these lines of code below into your config file.

   ```python
   from colossalai.amp import AMP_TYPE

   fp16 = dict(
     mode=AMP_TYPE.TORCH
   )
   ```

3. Tell Colossal-AI where your config file is when launch the distributed environment. For example, the config file is in
the current directory.

   ```python
   import colossalai

   colossalai.launch(config='./config.py', ...)
   ```

In this way, Colossal-AI knows what features you want to use and will inject this feature during `colossalai.initialize`.

### Global Hyper-parameters

Besides feature specification, the config file can also serve as a place to define your training hyper-parameters. This
comes handy when you want to perform multiple experiments, each experiment details can be put into a single config file
to avoid confusion. These parameters will be stored in the global parallel context and can be accessed in the training script.

For example, you can specify the batch size in your config file.

```python
BATCH_SIZE = 32
```

After launch, you are able to access your hyper-parameters through global parallel context.

```python
import colossalai
from colossalai.core import global_context as gpc

colossalai.launch(config='./config.py', ...)

# access your parameter
print(gpc.config.BATCH_SIZE)

```

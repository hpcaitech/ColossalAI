# 构建配置文件

作者: Guangyang Lu, Shenggui Li, Siqi Mai

> ⚠️ 此页面上的信息已经过时并将被废弃。请在[Booster API](../basics/booster_api.md)页面查阅更新。

**预备知识:**
- [分布式训练](../concepts/distributed_training.md)
- [Colossal-AI 总览](../concepts/colossalai_overview.md)


## 简介

在 Colossal-AI 中，我们需要一个配置文件来指定系统在训练过程中要注入的特征。在本教程中，我们将向您介绍如何构建您的配置文件以及如何使用这个配置文件。使用配置文件有以下一些好处：

1. 您可以在不同的配置文件中存储您的特征配置和训练超参数。
2. 对于我们未来发布的新功能，您亦可以在配置中指定，而无需改变训练脚本的代码。

在本教程中，我们将向您介绍如何构建您的配置文件。

## 配置定义

在一个配置文件中，有两种类型的变量。一种是作为特征说明，另一种是作为超参数。所有与特征相关的变量都是保留关键字。例如，如果您想使用混合精度训练，需要在 config 文件中使用变量名`fp16`，并遵循预先定义的格式。

### 功能配置

Colossal-AI 提供了一系列的功能来加快训练速度。每个功能都是由配置文件中的相应字段定义的。在本教程中，我们不会给出所有功能的配置细节，而是提供一个如何指定一个功能的说明。**每个功能的细节可以在其各自的教程中找到。**

为了说明配置文件的使用，我们在这里使用混合精度训练作为例子。您需要遵循以下步骤。

1. 创建一个配置文件（例如 `config.py`，您可以指定任意的文件名）。
2. 在配置文件中定义混合精度的配置。例如，为了使用 PyTorch 提供的原始混合精度训练，您只需将下面这几行代码写入您的配置文件中。

   ```python
   from colossalai.amp import AMP_TYPE

   fp16 = dict(
     mode=AMP_TYPE.TORCH
   )
   ```

3. 当启动分布式环境时，向 Colossal-AI 指定您的配置文件的位置。比如下面的例子是配置文件在当前目录下。

   ```python
   import colossalai

   colossalai.launch(config='./config.py', ...)
   ```

这样，Colossal-AI 便知道您想使用什么功能，并会在 `colossalai.initialize` 期间注入您所需要的功能。

### 全局超参数

除了功能的配置，您还可以在配置文件中定义训练的超参数。当您想进行多个实验时，这将会变得非常方便。每个实验的细节都可以放在独立的配置文件中，以避免混乱。这些参数将被存储在全局并行环境中，可以在训练脚本中访问。

例如，您可以在配置文件中指定批量大小。

```python
BATCH_SIZE = 32
```

启动后，您能够通过全局并行上下文访问您的超参数。

```python
import colossalai
from colossalai.core import global_context as gpc

colossalai.launch(config='./config.py', ...)

# access your parameter
print(gpc.config.BATCH_SIZE)

```

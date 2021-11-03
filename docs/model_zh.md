# 定义符合您需求的并行模型

如果您在训练一个拥有数亿级参数的巨大MLP模型，那么该模型一定无法在单个GPU上直接进行训练，不用担心，Colossal-AI可以帮您解决这一问题。您仍旧可以像写单GPU模型那样来写您的模型，Colossal-AI会按照您的并行设置自动将模型参数进行切割，并将它们均匀地存入一组GPU中。下面是一个简单的例子，来向您展示如何在Colossal-AI环境下写一个2D张量并行的模型。

## 简单的2D张量并行模型

```python
from colossalai.nn import Linear2D
import torch.nn as nn

class MLP_2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = Linear2D(in_features=1024, out_features=16384)
        self.linear_2 = Linear2D(in_features=16384, out_features=1024)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x
```

## 使用事先定义好的模型

为了您使用的方便，我们事先在我们的Model Zoo中定义好了一些现在流行的模型，比如*BERT*、*VIT*以及*MLP-Mixer*等，您可以根据您的需求来自定义这些模型的规模。

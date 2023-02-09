# 定义你自己的并行模型

作者: Zhengda Bian, Yongbin Li

> ⚠️ 我们正在编写此文档以使其更加详细。 我们将介绍不同并行的机制以及如何使用它们来编写模型。

假设您有一个具有数十亿参数的巨大 MLP 模型，其极大的隐藏层大小使其无法直接被单个 GPU 容纳。别担心，Colossal-AI 可以帮你解决这个问题。
在 Colossal-AI 的帮助下，您可以用所熟悉的为单个 GPU 编写模型的方式编写大模型，而 Colossal-AI 会自动拆分您的模型权重，并将它们完美地分配到一组 GPU 中。我们给出一个简单的示例，展示如何在 Colossal-AI 中编写简单的 2D 并行模型。

## 写一个简单的2D并行模型

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

## 使用预定义的模型

为了方便您的使用，我们在 Colossal-AI 的 Model Zoo 中提供一些流行的模型，如*BERT*, *ViT*, *MoE* 和 *GPT*，请自由地将它们定制为不同的尺寸，以满足您的特殊需求。

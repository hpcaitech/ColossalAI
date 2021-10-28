# Define your own parallel model

## Write a Simple 2D Parallel Model

Let's say we have a huge MLP model and its very large hidden size makes it difficult to fit into a single GPU. We can
then distribute the model weights across GPUs in a 2D mesh while you still write your model in a familiar way.

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

## Use pre-defined model
Our Model Zoo supports *BERT*, *VIT*, *MLP-Mixer* of different sizes.
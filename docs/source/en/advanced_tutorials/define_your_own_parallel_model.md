# Define your own parallel model

Author: Zhengda Bian, Yongbin Li

> ⚠️ We are working on this documentation to make it more detailed. We will introduce the mechanism of different parallelism
> and how to use them to write a model.

Let's say that you have a huge MLP model with billions of parameters and its extremely large hidden layer size makes it
impossible to fit into a single GPU directly. Don't worry, Colossal-AI is here to help you sort things out. With the help of Colossal-AI,
you can write your model in the familiar way in which you used to write models for a single GPU, while Colossal-AI automatically
splits your model weights and fit them perfectly into a set of GPUs. We give a simple example showing how to write a simple
2D parallel model in the Colossal-AI context.

## Write a simple 2D parallel model

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

For the sake of your convenience, we kindly provide you in our Model Zoo with some prevalent models such as *BERT*, *ViT*, *MoE*,
and *GPT*. Feel free to customize them into different sizes to fit into your special needs.

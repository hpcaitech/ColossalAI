# Parallelization

## Configure the Combination of Parallelization

We support multiple parallelization in our library.

Hybrid parallelism in our codebase, namely data parallelism, pipeline parallelism and tensor parallelism (
1D, 2D, 2.5D, 3D). You can initialize the corresponding process group by setting `parallel` in our config. The parallel
configuration can be easily deployed by a dictionary in configuration file. The configuration dictionary must obey the
following format. Data parallel size will be inferred automatically based on your inputs to pipeline parallelism and
tensor parallelism.

```python
parallel = dict(
    pipeline=dict("size": int),
    tensor=dict("size": int, "mode": '1d' or '2d' or '2.5d' or '3d', "kwargs": Any)
)
```

The name of the dictionary variable should be **parallel**. All the arguments even **parallel** itself are optional and data,
pipeline, tensor parallel size will be set to defaulted value 1. The value of data, pipeline and tensor can be a int
representing the size of specific parallel dimension or a dictionary with a key called "size". The key "mode"
represents the way of tensor parallelism.

## Data Parallel

Data parallel is the most common way to distribute your training task by splitting data into several shards and train 
on a single shard on each device. The configuration for data parallel is detected automatically and set for you. You do 
not have to explicitly set them in your configurations. When data parallel size is larger than 1, Colossal-AI automatically 
adds the distributed data sampler to the dataloader to shard the dataset.

## 1D, 2D, 2.5D and 3D Parallel

To enable hybrid parallelism, we provide an array of tensor parallelism. We provide the list of papers which match each 
tensor parallel method. These parallel modes need to work with the distributed layers provided by Colossal-AI.
- 1D: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- 2D: [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)  
2D parallel relies on the SUMMA matrix multiplication algorithm and splits the input data, 
model weights and layer outputs along two different dimensions. The tensor chunks are distributed over a 2D mesh of $P = N^2$ 
devices where $N$ is the number of tensor chunks in a single dimension.

- 2.5D: [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)  
Inspired by the 2.5D matrix multiplication algorithm, 2.5D parallel introduces a novel tensor parallelism which further 
parallelizes 2D tensor parallelism. An amount of $P = N^2 ∗ d$ processors are arranged into $d$ layers, 
where each layer performs matrix multiplication operations independently with a dimension $N$.

- 3D: [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)  
We also introduce a 3D tensor parallelism that parallelizes neural networks on a 3D processor cube. This method achieves 
the optimal, $O(P^{1/3})$ communication overhead on $P$ processors, while both computation and memory usage are evenly distributed 
through optimized load balancing of parameters as well as activations.

```python
# 1D parallel
parallel = dict(
    pipeline=dict(size=1), # number of pipeline stages
    tensor=dict(size=4, mode='1d')
)

# 2D parallel
parallel = dict(
    pipeline=dict(size=1), # number of pipeline stages
    tensor=dict(size=4, mode='2d')
)

# 2.5D parallel
parallel = dict(
    pipeline=dict(size=1), # number of pipeline stages
    tensor=dict(size=8, mode='2.5d', depth=2)
)

# 3D parallel
parallel = dict(
    pipeline=dict(size=1), # number of pipeline stages
    tensor=dict(size=8, mode='3d')
)
```

## Pipeline Parallel (experimental)

Pipeline parallelism is to split the model into several partitions by layer. For example, let's assume we have a simple 
model which consists of two linear layer. We have two GPUs, and we can allocate the first linear layer to the first GPU 
and the second layer to the second GPU. This example of course wastes the computing resources and is only to demonstrate
the idea of pipeline parallelism. 

As PyTorch is based on dynamic computation graph, the computation flow is not known until execution. To support pipeline 
parallelism in PyTorch, you may need to add one more attribute, `layers_cfg` in your model class which tells Colossal-AI
the sequence of execution. One example you can refer is `colossalai.nn.model.VanillaResNet`.

```python
from colossalai.nn import BaseModel
import torch

class VanillaResNet(BaseModel):

    def __init__(
            self,
            num_cls: int,
            block_type: str,
            layers: List[int],
            norm_layer_type: str = 'BatchNorm2d',
            in_channels: int = 3,
            groups: int = 1,
            width_per_group: int = 64,
            zero_init_residual: bool = False,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            dilations=(1, 1, 1, 1)
    ) -> None:
        super().__init__()
        
        ... # some model params
        
        self.layers_cfg = [
            # conv1
            dict(type='Conv2d',
                 in_channels=in_channels,
                 out_channels=self.inplanes,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 bias=False),
            # bn1
            dict(
                type=norm_layer_type,
                num_features=self.inplanes
            ),
            # relu
            dict(
                type='ReLU',
                inplace=True
            ),
            # maxpool
            dict(
                type='MaxPool2d',
                kernel_size=3,
                stride=2,
                padding=1
            ),
            # layer 1
            dict(
                inplanes=self.inplanes,
                planes=64,
                blocks=self.blocks[0],
                dilation=self.dilations[0],
                **self.reslayer_common_cfg
            ),
            # layer 2
            dict(
                inplanes=64 * self.block_expansion,
                planes=128,
                blocks=self.blocks[1],
                stride=2,
                dilate=replace_stride_with_dilation[0],
                dilation=self.dilations[1],
                **self.reslayer_common_cfg
            ),
            # layer  3
            dict(
                inplanes=128 * self.block_expansion,
                planes=256,
                blocks=layers[2],
                stride=2,
                dilate=replace_stride_with_dilation[1],
                dilation=self.dilations[2],
                **self.reslayer_common_cfg
            ),
            # layer 4
            dict(
                inplanes=256 * self.block_expansion,
                planes=512,
                blocks=layers[3], stride=2,
                dilate=replace_stride_with_dilation[2],
                dilation=self.dilations[3],
                **self.reslayer_common_cfg
            ),
            # avg pool
            dict(
                type='AdaptiveAvgPool2d',
                output_size=(1, 1)
            ),
            # flatten
            dict(
                type='LambdaWrapper',
                func=lambda mod, x: torch.flatten(x, 1)
            ),
            # linear
            dict(
                type='Linear',
                in_features=512 * self.block_expansion,
                out_features=num_cls
            )
        ]
```

You can set the number of pipeline stages in your configuration file. When pipeline size is larger than 1, Colossal-AI 
will automatically creates the pipeline schedule which defines the forward and backward step. You can specify how many microbatches
to run in each step in the `schedule` configuration.

```python
parallel = dict(
    pipeline=dict(size=1), # number of pipeline stages
    tensor=dict(size=1, mode=None)
)

schedule = dict(
    num_microbatches = 4 # set the number of microbatches per step
)
```
This feature is still in development and is only experimental for now.

## Sequence Parallel (experimental)

Sequence parallel is to support long-sequence modelling such as document-level text understanding and medical imaging. 
This method is proposed in [Sequence Parallelism: Making 4D Parallelism Possible](https://arxiv.org/abs/2105.13120). 
This feature is still in development and is only experimental for now.

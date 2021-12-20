# Parallelization

## Configure the Combination of Parallelization

We support multiple parallelization in our library.

Hybrid parallelism in our codebase refers to namely the combination of data parallelism, pipeline parallelism 
and tensor parallelism (1D, 2D, 2.5D, 3D). Each parallelism requires different network topology and thus 
different initializers for distributed process group. You can initialize the corresponding process group by 
setting `parallel` in our config. The parallel configuration can be easily deployed by a dictionary in 
configuration file. The configuration dictionary must obey the following format. Data parallel size will be 
inferred automatically based on your inputs to pipeline parallelism and tensor parallelism. The distributed 
environment will set up by `colossalai.launch`.

```python
# sampler format
parallel = dict(
    pipeline=dict("size": int),
    tensor=dict("size": int, "mode": '1d' or '2d' or '2.5d' or '3d', "kwargs": Any)
)

# this is ok
parallel = dict(
    pipeline=dict(size=2),
    tensor=dict(size=4, mode='2d')
)

# this is ok
parallel = dict(
    pipeline=2,
    tensor=dict(size=4, mode='2d')
)

# this is not ok
# as you need to specify the mode for tensor parallelism
parallel = dict(
    pipeline=2,
    tensor=4
)

# this is ok as well as tensor will be default to size 1 
# and mode None
parallel = dict(
    pipeline=2
)

# this is ok as well as pipeline will default to size 1
parallel = dict(
    tensor=dict(size=4, mode='2d')
)

```

The name of the dictionary variable should be **parallel**. All the arguments even **parallel** itself are optional and
data, pipeline, tensor parallel size will be set to defaulted value 1. The value of data, pipeline and tensor can be a
int representing the size of specific parallel dimension or a dictionary with a key called "size". The key "mode"
represents the way of tensor parallelism. 

**You can choose to not have 'parallel' in your configuration and both pipelineand tensor will default to size 1.**


## Data Parallel

Data parallel is the most common way to distribute your training task by splitting data into several shards and train on
a single shard on each device. The configuration for data parallel is detected automatically and set for you. You do not
have to explicitly set them in your configurations. There are two ways to handle the all-reduce in data parallel in Colossal-AI.

1. If you specify gradient handlers, gradients will be all-reduced according to the gradient handlers
2. Otherwise, PyTorch DistributedDataParallel will be used

In most cases, you will be using the second mode unless you have complex handling of the gradients.

## 1D, 2D, 2.5D and 3D Parallel

To enable hybrid parallelism, we provide an array of tensor parallelism. We provide the list of papers which match each
tensor parallel method. These parallel modes need to work with the distributed layers provided by Colossal-AI.

- 1D: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- 2D: [An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)  
  2D parallel relies on the SUMMA matrix multiplication algorithm and splits the input data, model weights and layer
  outputs along two different dimensions. The tensor chunks are distributed over a 2D mesh of $P = N^2$ devices where
  $N$ is the number of tensor chunks in a single dimension.

- 2.5D: [2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)  
  Inspired by the 2.5D matrix multiplication algorithm, 2.5D parallel introduces a novel tensor parallelism which
  further parallelizes 2D tensor parallelism. An amount of $P = N^2 ∗ d$ processors are arranged into $d$ layers, where
  each layer performs matrix multiplication operations independently with a dimension $N$.

- 3D: [Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)  
  We also introduce a 3D tensor parallelism that parallelizes neural networks on a 3D processor cube. This method
  achieves the optimal, $O(P^{1/3})$ communication overhead on $P$ processors, while both computation and memory usage
  are evenly distributed through optimized load balancing of parameters as well as activations.

```python
# 1D parallel
parallel = dict(
    tensor=dict(size=4, mode='1d')
)

# 2D parallel
parallel = dict(
    tensor=dict(size=4, mode='2d')
)

# 2.5D parallel
parallel = dict(
    tensor=dict(size=8, mode='2.5d', depth=2)
)

# 3D parallel
parallel = dict(
    tensor=dict(size=8, mode='3d')
)
```

Once you specify the tensor parallel mode in your configuration, you can proceed to use its corresponding distributed 
operator. For example, if you mode is '2d', you can use `colossalai.nn.Linear2D` in you model construction.


## Pipeline Parallel (experimental)

Pipeline parallelism is to split the model into several partitions by layer. For example, let's assume we have a simple
model which consists of two linear layer. We have two GPUs, and we can allocate the first linear layer to the first GPU
and the second layer to the second GPU. 

You can set the number of pipeline stages in your configuration file. When pipeline size is larger than 1, Colossal-AI
will automatically creates the pipeline schedule which defines the forward and backward step. 

```python
parallel = dict(
    pipeline=dict(size=4), # number of pipeline stages
)
```

As PyTorch is based on dynamic computation graph, the computation flow is not known until execution. To support pipeline parallelism, you have the following two ways to split your model,

1. Split your model directly. Below is an exmaple of resnet split into two pipeline stages.
```python
from torchvision.models import resnet18
from colossalai.core import global_context as gpc

model = resnet18(num_classes=10)

if gpc.get_local_rank(ParallelMode.PIPELINE) == 0:
    model = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2
    )
elif gpc.get_local_rank(ParallelMode.PIPELINE) == 1:
    from functools import partial

    class Flatten(nn.Module):

        def forward(self, x):
            return torch.flatten(x, 1)

    model = nn.Sequential(
        model.layer3,
        model.layer4,
        model.avgpool,
        Flatten(),
        model.fc
    )
```


2. Make sure your model inherit `colossalai.nn.model.ModelFromConfig` and registered into the
`MODELS` registry. Define the `self.layers_cfg` attribute. 
Pass in a dict/Config object which specifies the parameters of your model. 
Use `colossalai.builder.pipeline.build_pipeline_model_from_cfg` to partition the layers.

```python
from colossalai.builder import build_pipeline_model_from_cfg
from colossalai.nn.model import ModelFromConfig
from colossalai.registry import MODELS


@MODELS.register_module
class MyModel(ModelFromConfig):

    def __init__(self, arg1, arg2, ...):
        ...
        self.layers_cfg = [
            dict(type='Linear', in_features=3, out_features=512),
            dict(type='Linear', in_features=512, out_features=512),
            ...
        ]


model_cfg = dict(
    type='MyModel',
    arg1=1,
    arg2=2
    ...
)

# from config 
model = build_pipeline_model_from_cfg(model_cfg, num_chunks=1)

# from torch.nn.Sequential
# model = build_pipeline_model(sequential_model, num_model_chunks)

```

When your model is split into partitions, you can use PipelineSchedule to execute training.

```python
import colossalai
from colossalai.engine.schedule import PipelineSchedule

engine, train_dataloader, _, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader) 

schedule = PipelineSchedule(num_microbatches=4)

# interleaved pipeline
# schedule = InterleavedPipelineSchedule(num_microbatches=4, num_model_chunks=2)

# execute a training epoch
data_iter = iter(train_dataloader)

for i in range(len(train_dataloader)):
    output, label, loss = schedule.forward_backward_step(engine,
                                                        data_iter,
                                                        forward_only=False,
                                                        )

```

This feature is still in development and is only experimental for now.

## Sequence Parallel (experimental)

Sequence parallel is to support long-sequence modelling such as document-level text understanding and medical imaging.
This method is proposed in [Sequence Parallelism: Making 4D Parallelism Possible](https://arxiv.org/abs/2105.13120).
This feature is still in development and is only experimental for now.

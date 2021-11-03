# 并行技术

## 配置并行技术组合

ColossalAI支持多种并行技术，包括数据并行、张量并行（1D、2D、2.5D、3D）、流水线并行以及序列并行。您可以通过更改配置文件中的`parallel`字典变量
来初始化分布式系统中的进程组，配置文件中的`parallel`字典变量必须满足下面的格式。数据并行的规模可以通过`parallel`中流水线并行的规模和张量并行的
规模计算得出。

```python
parallel = dict(
    pipeline=dict("size": int),
    tensor=dict("size": int, "mode": '1d' or '2d' or '2.5d' or '3d', "kwargs": Any)
)
```

注意该字典变量的名称必须为**parallel**。该变量中所有的参数，包括`parallel`本身都是非必需的，如果您的代码中没有提供该变量，则所有并行规模都将被
设定为默认值1，即不使用任何并行技术的情况。`parallel`中data、pipeline以及tensor的值分别代表了数据并行、流水线并行、以及张量并行的规模，而mode
的值代表了张量并行的模式。

## 数据并行

数据并行是一种最常见的并行技术，可以将数据分成几个不同的部分，并对每一个部分在一台设备上进行训练。ColossalAI可以自动检测数据并行设置并为您设置好环境，
您不需要在您的环境配置中显式地设置。当数据并行规模大于1时，ColossalAI会自动为数据读取器增加分布式数据采样器，以此来达到切分数据集的目的。

## 1D、2D、2.5D与3D张量并行

为了方便混合并行技术，我们提供了一系列的张量并行技术，同时下面罗列了每一种张量并行技术对应的论文，这些张量并行技术需要ColossalAI提供的分布式层结构的支持。
- 1D：[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- 2D：[An Efficient 2D Method for Training Super-Large Deep Learning Models](https://arxiv.org/abs/2104.05343)
2D张量并行依赖SUMMA矩阵乘法技术，其在两个不同的维度上对于输入数据进行切分。切分后的张量分布在一个的2D网格上，使用的总设备数量为$P = N^2$，其中$N$为
一个维度上的切分张量数量。

- 2.5D：[2.5-dimensional distributed model training](https://arxiv.org/abs/2105.14500)
2.5D并行技术受到了2.5D矩阵乘法的启发，其对于2D张量并行的结果进行进一步切分，在$d$层上面安排$P = N^2 ∗ d$个处理器，相应地，矩阵乘法操作也被切分为$d$份
在不同的层上面进行。

- 3D：[Maximizing Parallelism in Distributed Training for Huge Neural Networks](https://arxiv.org/abs/2105.14450)
我们还引入了3D张量并行技术，该技术在一个3D处理器立方体中对神经网络参数进行并行化。使用$P$个处理器时，该并行技术可以在付出$O(P^{1/3})$的通信开销的情况下
达到最优表现，且计算资源和内存使用都可以在$P$个处理器上达到平均分配。

使用上述几种张量并行的`parallel`字典变量示例参见下方代码。

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

## 流水线并行（开发中）

流水线并行指的是在将深度学习模型按照层切分为几个不同的部分，例如，假设一个由两个线性层组成的简单模型，我们可以使用两个GPU，那么我们可以把第一个线性层
的工作分配给一个GPU，把第二个线性层的工作分配给另一个GPU。当然这个例子只是为了说明流水线并行的工作方式，没有实际意义。

由于PyTorch的计算基于动态计算图，所以在执行前无法确定计算流。为了支持PyTorch中的流水线并行，您需要为您的模型类加入一个额外的特征`layers_cfg`，
使ColossalAI清楚具体的计算流程，`colossalai.nn.VanillaResNet`给出了一个您可以参考的示例。

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

您可以在配置文件中手动设置流水线并行的级数，当柳树线并行级数大于1时，ColossalAI将会自动创建定义前向传播和后向传播的流水线调度程序。同时您还可以在配置文件
中的`schedule`字典变量来定义每一个步骤中训练的微批数量。下面的代码给出了一个配置流水线并行的例子。

```python
parallel = dict(
    pipeline=dict(size=1), # number of pipeline stages
    tensor=dict(size=1, mode=None)
)

schedule = dict(
    num_microbatches = 4 # set the number of microbatches per step
)
```
目前该并行技术仍处于实验开发阶段。

## 序列并行（开发中）

序列并行是为了支持对于长序列数据的建模，这类数据包括文档级别的文本理解以及医学影像分析，该并行技术由论文
[Sequence Parallelism: Making 4D Parallelism Possible](https://arxiv.org/abs/2105.13120)提出。
目前该并行技术仍处于实验开发阶段。

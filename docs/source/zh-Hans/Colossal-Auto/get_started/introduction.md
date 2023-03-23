# 介绍

近年来，大规模机器学习模型的部署受到越来越多的重视。然而，目前常见的分布式大模型训练方案，都依赖用户**人工反复尝试**和系统专家的经验来进行配置部署。这对绝大多数AI开发者来说十分不友好，因为他们不希望将时间精力花费在研究分布式系统和试错上。
Colossal-AI的**Colossal-Auto** 帮助AI开发者简化了大规模机器学习模型的部署过程。相比现有其他手动配置复杂并行策略和修改模型的解决方案，Colossal-Auto 仅需增加一行代码，提供 cluster 信息以及单机训练模型即可获得分布式训练能力，并且**原生支持包括 Hugging Face，Timm 等热门 AI 模型库**。



## 概览

<figure style={{textAlign: "center"}}>
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/auto_parallel/auto_parallel.png"/>
</figure>

## 用法
```python
# wrap the model using auto_engine
model = autoparallelize(model, meta_input_samples)
# normal training loop
...
```


## 图追踪
Colossal-Auto 是**首个基于 PyTorch 框架使用静态图分析的自动并行系统**。PyTorch 作为一个动态图框架，获取其静态的执行计划是机器学习系统领域被长期研究的问题。Colossal-Auto 使用基于 torch.FX Tracer 的 ColoTracer 来完成对于最优并行策略的搜索。在 tracing 过程中推导并记录了每个 tensor 的元信息，例如 tensor shape，dims，dtype 等。因此 Colossal-AI 具有更好的模型泛化能力，而不是依靠模型名或手动修改来适配并行策略。


## 细粒度分布式训练策略搜索

我们调研了很多现有的自动并行系统（<a href="https://arxiv.org/abs/1807.08887"> Tofu </a>, <a href="https://arxiv.org/abs/1807.05358"> Flexflow </a>, <a href="https://arxiv.org/abs/2201.12023"> Alpa </a>），以及自动激活值检查点算法（<a href="https://hal.inria.fr/hal-02352969"> Rotor </a>, <a href="https://arxiv.org/abs/1604.06174"> Sublinear </a>），在他们的启发下，我们开发一个基于PyTorch框架的自动并行系统Colossal-Auto。Colossal-Auto会在满足内存预算的限制下，以最快运行时间为目标，为每个 op 进行策略搜索，最终得到真实训练时的策略，包括每个 tensor 的切分策略，不同计算节点间需要插入的通信算子类型，是否要进行算子替换等。现有系统中的张量并行，数据并行，NVIDIA 在 Megatron-LM 等并行系统中使用的 column 切分和 row 切分并行等混合并行，都是自动并行可以搜索到的策略的子集。除了这些可以手动指定的并行方式外，Colossal-AI 有能力为每个 op 指定独特的并行方式，因此有可能找到比依赖专家经验和试错配置的手动切分更好的并行策略。



## 分布式 tensor 与 shape consistency 系统

与 PyTorch 最新发布的 DTensor 类似，Colossal-AI 也使用了 device mesh 对集群进行了抽象管理。具体来说，Colossal-AI 使用 sharding spec 对 tensor 的分布式存储状态进行标注，使用 shape consistency manager 自动地对同一 tensor 在不同 sharding spec 间进行转换。这让 Colossal-AI 的通用性和易用性极大地提升，借助 shape consistency manager 可以没有负担地切分 tensor，而不用担心上游 op 的 output 与下游的 input 在集群中的存储方式不同。


相较于 PyTorch DTensor，Colossal-AI 有以下优势：
+ Colossal-AI 的 device mesh 可以 profiling 到集群性能指标，对不同的通信算子进行耗时估算。
+ Colossal-AI 的 shape consistency 会贪心地搜索 sharding spec 间的转换方式，而不是朴素地逐 dimension 进行转换，这样能找到更高效的转换路径，进而使得 sharding spec 间的转换通信开销更小。
+ 加入了 all_to_all 操作，使得 Colossal-AI 的扩展性更强，这在大规模集群上进行训练时，可以展现出很大的优势。

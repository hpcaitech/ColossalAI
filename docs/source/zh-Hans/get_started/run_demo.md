# 快速演示

Colossal-AI 是一个集成的大规模深度学习系统，具有高效的并行化技术。该系统可以通过应用并行化技术在具有多个 GPU 的分布式系统上加速模型训练。该系统也可以在只有一个 GPU 的系统上运行。以下是展示如何使用 Colossal-AI 的 Quick demos。

## 单 GPU

Colossal-AI 可以用在只有一个 GPU 的系统上训练深度学习模型，并达到 baseline 的性能。 我们提供了一个 [在 CIFAR10 数据集上训练 ResNet](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/resnet) 的例子，该例子只需要一个 GPU。
您可以在 [ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples) 中获取该例子。详细说明可以在其 `README.md` 中获取。

## 多 GPU

Colossal-AI 可用于在具有多个 GPU 的分布式系统上训练深度学习模型，并通过应用高效的并行化技术大幅加速训练过程。我们提供了多种并行化技术供您尝试。

#### 1. 数据并行

您可以使用与上述单 GPU 演示相同的 [ResNet 例子](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/resnet)。 通过设置 `--nproc_per_node` 为您机器上的 GPU 数量，您就能把数据并行应用在您的例子上了。

#### 2. 混合并行

混合并行包括数据、张量和流水线并行。在 Colossal-AI 中，我们支持不同类型的张量并行（即 1D、2D、2.5D 和 3D）。您可以通过简单地改变 `config.py` 中的配置在不同的张量并行之间切换。您可以参考 [GPT example](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt), 更多细节能在它的 `README.md` 中被找到。

#### 3. MoE 并行

<!-- TODO: 在colossalai中实现这个例子 -->

我们提供了一个 [ViT-MoE 例子](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/moe) 来验证 MoE 的并行性。 WideNet 使用 Mixture of Experts（MoE）来实现更好的性能。更多的细节可以在我们的教程中获取：[教会您如何把 Mixture of Experts 整合到模型中](../advanced_tutorials/integrate_mixture_of_experts_into_your_model.md)。

#### 4. 序列并行

序列并行是为了解决 NLP 任务中的内存效率和序列长度限制问题。 我们在 [ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples) 中提供了一个 [Sequence Parallelism 例子](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/sequence_parallel)。您可以按照 `README.md` 来执行代码。

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 run_demo.py  -->

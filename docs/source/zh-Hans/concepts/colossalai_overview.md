# Colossal-AI 总览

作者: Shenggui Li, Siqi Mai

## 关于 Colossal-AI

随着深度学习模型规模的发展，向新的训练模式转变是非常重要的。没有并行和优化的传统训练方法将成为过去，新的训练方法是使训练大规模模型高效和节省成本的关键。

Colossal-AI 是一个集成的系统，为用户提供一套综合的训练方法。您可以找到常见的训练方法，如混合精度训练和梯度累积。此外，我们提供了一系列的并行技术，包括数据并行、张量并行和流水线并行。我们通过不同的多维分布式矩阵乘法算法来优化张量并行。我们还提供了不同的流水线并行方法，使用户能够有效地跨节点扩展他们的模型。更多的高级功能，如卸载，也可以在这个教程文档中找到详细的内容。

## Colossal-AI 的使用

我们的目标是使 Colossal-AI 易于使用，并且对用户的代码不产生干扰。如果您想使用Colossal-AI，这里有一个简单的一般工作流程。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/01/28/ZK7ICWzbMsVuJof.png"/>
<figcaption>Workflow</figcaption>
</figure>

1. 准备一个配置文件，指定您要使用的功能和参数。
2. 用 `colossalai.launch` 初始化分布式后端。
3. 用 `colossalai.booster` 将训练特征注入您的训练组件（如模型、优化器）中。
4. 进行训练和测试.

我们将在`基本教程`部分介绍整个工作流程。

## 未来计划

Colossal-AI 系统将会进一步拓展和优化，包括但不限于:

1. 分布式操作的优化
2. 异构系统训练的优化
3. 从模型大小的维度切入，提升训练速度并维持精度
4. 拓展现有的并行方法

**我们始终欢迎社区的建议和讨论，如果您遇到任何问题，我们将非常愿意帮助您。您可以在GitHub 提 [issue](https://github.com/hpcaitech/ColossalAI/issues) ，或在[论坛](https://github.com/hpcaitech/ColossalAI/discussions)上创建一个讨论主题。**

<!-- doc-test-command: echo "colossalai_overview.md does not need test"  -->

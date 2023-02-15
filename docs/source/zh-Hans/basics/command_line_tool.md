# 命令行工具

作者: Shenggui Li

**预备知识:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

## 简介

Colossal-AI给用户提供了命令行工具，目前命令行工具可以用来支持以下功能。
- 检查Colossal-AI是否安装正确
- 启动分布式训练
- 张量并行基准测试

## 安装检查

用户可以使用`colossalai check -i`这个命令来检查目前环境里的版本兼容性以及CUDA Extension的状态。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/05/04/KJmcVknyPHpBofa.png"/>
<figcaption>Check Installation Demo</figcaption>
</figure>

## 启动分布式训练

在分布式训练时，我们可以使用`colossalai run`来启动单节点或者多节点的多进程，详细的内容可以参考[启动 Colossal-AI](./launch_colossalai.md)。

## 张量并行基准测试

Colossal-AI提供了多种张量并行，想要充分理解这些方法需要一定的学习成本，对于新手来说很难靠经验选择一个并行方式。
所以我们提供了一个简单的基准测试，能够让用户在自己的机器上测试不同张量并行的性能。这个基准测试跑一个并行的MLP模型，
输入数据的维度为`（批大小，序列长度，隐藏层维度）`。通过指定GPU的数量，Colossal-AI会搜索所有可行的并行配置。用户可以通过查看`colossalai benchmark --help`来自定义相关的测试参数。

```shell
# 使用4个GPU
colossalai benchmark --gpus 4

# 使用8个GPU
colossalai benchmark --gpus 8
```

:::caution

目前仅支持单节点的基准测试。

:::

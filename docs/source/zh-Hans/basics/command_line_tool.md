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

<!-- doc-test-command: echo  -->

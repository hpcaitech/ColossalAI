# 集群实用程序

作者: [Hongxin Liu](https://github.com/ver217)

**前置教程:**
- [分布式训练](../concepts/distributed_training.md)

## 引言

我们提供了一个实用程序类 `colossalai.cluster.DistCoordinator` 来协调分布式训练。它对于获取有关集群的各种信息很有用，例如节点数、每个节点的进程数等。

## API 参考

{{ autodoc:colossalai.cluster.DistCoordinator }}

{{ autodoc:colossalai.cluster.DistCoordinator.is_master }}

{{ autodoc:colossalai.cluster.DistCoordinator.is_node_master }}

{{ autodoc:colossalai.cluster.DistCoordinator.is_last_process }}

{{ autodoc:colossalai.cluster.DistCoordinator.print_on_master }}

{{ autodoc:colossalai.cluster.DistCoordinator.print_on_node_master }}

{{ autodoc:colossalai.cluster.DistCoordinator.priority_execution }}

{{ autodoc:colossalai.cluster.DistCoordinator.destroy }}

{{ autodoc:colossalai.cluster.DistCoordinator.block_all }}

{{ autodoc:colossalai.cluster.DistCoordinator.on_master_only }}

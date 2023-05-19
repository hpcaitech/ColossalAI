# Cluster Utilities

Author: [Hongxin Liu](https://github.com/ver217)

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)

## Introduction

We provide a utility class `colossalai.cluster.DistCoordinator` to coordinate distributed training. It's useful to get various information about the cluster, such as the number of nodes, the number of processes per node, etc.

## API Reference

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

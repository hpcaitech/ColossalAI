# Cluster Utilities

Author: [Hongxin Liu](https://github.com/ver217)

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)

## Introduction

We provide a utility class `colossalai.cluster.DistCoordinator` to coordinate distributed training. It's useful to get various information about the cluster, such as the number of nodes, the number of processes per node, etc.

## API Reference

{{ autodoc:colossalai.cluster.DistCoordinator }}

<!-- doc-test-command: echo  -->

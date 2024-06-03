# Colossal-AUTO

## Challenges
Recently, large models have achieved the state of the art performances in various fields. In order to support large model training, we have to use distributed training techniques. However, finding an efficient distributed execution plan not only requires fine-grained model statistics, such as memory and computing overhead of each operator but also is a labor-intensive task even for an expert in the field of distributed training.

## Our solution
To simplify the process of distributed training for foundational models, recent advancements in machine learning systems have led to the emergence of automatic parallel systems. We investigate and research a number of current automatic parallel systems(<a href="https://arxiv.org/abs/1807.08887"> Tofu </a>, <a href="https://arxiv.org/abs/1807.05358"> Flexflow </a>, <a href="https://arxiv.org/abs/2201.12023"> Alpa </a>) and some auto activation checkpoint algorithms(<a href="https://hal.inria.fr/hal-02352969"> Rotor </a>, <a href="https://arxiv.org/abs/1604.06174"> Sublinear </a>). Inspired from these advanced systems, we build an automatic parallel system upon PyTorch framework. The input of the system is the serial PyTorch code, and the output is a PyTorch program with an optimized distributed execution plan. It is worth emphasizing that the output is a regular PyTorch program, so it is compatible with runtime optimization methods, such as ZeRO-Offload and PatrickStar.

## Key modules

### Analyzer

**Analyzer** is a static analysis system consisting of three parts:
A *symbolic profiler* for collecting computing and memory overhead related to static computation graph, a *cluster detector* for collecting hardware characteristics and detecting cluster topology and a *tensor layout manager* to find efficient tensor layout conversion path from different sharding spec and record conversion cost.

### Solver

**Solver** is designed to find the optimal execution plan for a given computation graph and cluster in two stages:
1) *Intra-op parallelism stage* is to find the plan with the minimum total execution time of all nodes with respect to the constraint of the memory budget. The optimization goal of intra-op parallelism solver is modified from <a href="https://arxiv.org/abs/2201.12023"> Alpa </a>'s intra-op parallelism ILP solver.
2) *Activation checkpoint stage* is to search for the fastest execution plan that meets the memory budget on the computation graph after inserting the communication nodes by the intra-op parallelism stage. The algorithm to find optimal activation checkpoint is modified from <a href="https://hal.inria.fr/hal-02352969"> Rotor </a>. The reason we use two-stage optimization is that if the two tasks are formulated together, the solving time will be significantly increased, which will greatly affect the user experience of the system. On the contrary, solving in two hierarchical levels has many advantages. Firstly, compared with the computation graph with activation checkpointing, the original graph has fewer nodes, which can reduce the solving cost of intra-op parallelism solver. In addition, a more optimal solution can be found by adding the communication overhead into the activation checkpoint modeling.

### Generator
**Generator** applies the searched execution plan to the computation graph and recompiles the computation graph to optimized PyTorch code. It has *a series compile pass* to insert a communication node or do the kernel substitution as the intra-op parallelism solver required. Additionally, we implement a *code generation* feature to recognize the annotation from the activation checkpoint solver and inject the activation checkpoint block following annotation instructions.

# Quick Demo

Colossal-Auto simplifies the process of deploying large-scale machine learning models for AI developers. Compared to other solutions that require manual configuration of complex parallel policies and model modification, Colossal-Auto only requires one line of code from the user, along with cluster information and model configurations, to enable distributed training. Quick demos showing how to use Colossal-Auto are given below.

### 1. Basic usage

Colossal-Auto can be used to find a hybrid SPMD parallel strategy includes data, tensor(i.e., 1D, 2D, sequential) for each operation. You can follow the [GPT example](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt/experiments/auto_parallel).
Detailed instructions can be found in its `README.md`.

### 2. Integration with activation checkpoint

Colossal-Auto's automatic search function for activation checkpointing finds the most efficient checkpoint within a given memory budget, rather than just aiming for maximum memory compression. To avoid a lengthy search process for an optimal activation checkpoint, Colossal-Auto has implemented a two-stage search process. This allows the system to find a feasible distributed training solution in a reasonable amount of time while still benefiting from activation checkpointing for memory management. The integration of activation checkpointing in Colossal-AI improves the efficiency and effectiveness of large model training. You can follow the [Resnet example](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/auto_parallel).
Detailed instructions can be found in its `README.md`.

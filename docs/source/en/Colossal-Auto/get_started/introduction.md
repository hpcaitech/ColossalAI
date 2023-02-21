# Introduction

In recent years, the deployment of large-scale machine learning models has become increasingly important. However, distributed training systems often require **manual parallelization plans**, which can be complex and require expert knowledge in system engineering and configuration. This can be a challenge for most AI developers without the necessary skills. The need for manual parallelization can make deploying large-scale machine learning models difficult and expensive.

**Colossal-Auto** simplifies the process of deploying large-scale machine learning models for AI developers. Compared to other solutions that require manual configuration of complex parallel policies and model modification, Colossal-Auto only requires one line of code from the user, along with cluster information and model configurations, to enable distributed training. Technically, It seamlessly **integrates with popular AI model frameworks like Hugging Face and Timm.**



## Overview

<figure style={{textAlign: "center"}}>
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/auto_parallel/auto_parallel.png"/>
</figure>


## Usage

```python
# wrap the model using auto_engine
model = autoparallelize(model, meta_input_samples)
# normal training loop
...
```


## Graph Tracing

Colossal-Auto is **the first auto-parallelism system** that uses static graph analysis based on the PyTorch framework. Obtaining a static execution plan for PyTorch, a dynamic graph framework, has long been an area of research in the field of machine learning systems. Colossal-Auto uses ColoTracer, a forked version of the torch.FX Tracer, to guide the search for an optimal parallelization strategy. The meta-information of each tensor, such as tensor shape, dims, dtype, etc., is computed and recorded during the tracing process. This approach has the advantage of better generalization, as it is not tied to specific models or configurations.



## Fine-grained Parallelism Search
We investigate and research a number of current automatic parallel systems(<a href="https://arxiv.org/abs/1807.08887"> Tofu </a>, <a href="https://arxiv.org/abs/1807.05358"> Flexflow </a>, <a href="https://arxiv.org/abs/2201.12023"> Alpa </a>) and some auto activation checkpoint algorithms(<a href="https://hal.inria.fr/hal-02352969"> Rotor </a>, <a href="https://arxiv.org/abs/1604.06174"> Sublinear </a>). Inspired from these advanced systems, we build Colossal-Auto which is an automatic parallel system upon PyTorch framework. Colossal-Auto searches for strategies in regard to each operand with the goal of achieving the fastest runtime while meeting memory budget constraints. It ultimately determines the actual training time strategy, including the tensor split strategy for each tensor, the type of communication operators to be inserted between different computing nodes, whether to replace operators, etc. The tensor, data, and hybrid parallelism such as column and row split used by NVIDIA in Megatron-LM and other parallelism systems are all subsets of strategies that can be searched by Colossal-AI. In addition to these parallelisms that can be manually specified, Colossal-AI can specify a unique parallelism method for each operation and, potentially finding a better parallelism strategy than what human experts could provide.



## Distributed Tensor and Shape-Consistency System

The Colossal-AI system uses a device-mesh, similar to PyTorch's latest DTensor release, to manage its cluster. Colossal-AI uses a sharding-spec to annotate the storage status of each tensor and facilitate their distribution across the cluster. The system also employs a shape-consistency manager to automatically transform tensors between different sharding-specs, allowing for seamless slicing and dicing of tensors, while the shape-consistency manager ensures that the output of upstream operands is consistently stored in the cluster, regardless of how the input of downstream operands is stored. This makes Colossal-AI highly versatile and easy to use without users worrying about the storage status of tensors when performing operations on them.

Here are some key advantages of Colossal-AI compared to PyTorch DTensor:
Colossal-AI's device-mesh uses cluster performance metrics and profiling results to estimate the time consumption of different communication operators. This helps Colossal-AI optimize communication between nodes and improve overall system efficiency.
Colossal-AI's shape-consistency manager uses a greedy search algorithm to find relatively efficient ways to transform tensors between different sharding-specs, rather than simply transforming dimensions one by one. This can lead to more efficient and effective transformations.
The integration of all-to-all operations in Colossal-AI increases the scalability of the system by enabling more efficient communication between nodes. This is especially useful for large-scale machine learning tasks that require the transfer of large amounts of data between nodes.

# Quick Demo

Colossal-AI is an integrated large-scale deep learning system with efficient parallelization techniques. The system can
accelerate model training on distributed systems with multiple GPUs by applying parallelization techniques. The system
can also run on systems with only one GPU. Quick demos showing how to use Colossal-AI are given below.

## Single GPU

Colossal-AI can be used to train deep learning models on systems with only one GPU and achieve baseline
performances. We provided an example to [train ResNet on CIFAR10 dataset](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/resnet)
with only one GPU. You can find the example in [ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples).
Detailed instructions can be found in its `README.md`.

## Multiple GPUs

Colossal-AI can be used to train deep learning models on distributed systems with multiple GPUs and accelerate the
training process drastically by applying efficient parallelization techniques. When we have several parallelism for you to try out.

#### 1. data parallel

You can use the same [ResNet example](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/resnet) as the
single-GPU demo above. By setting `--nproc_per_node` to be the number of GPUs you have on your machine, the example
is turned into a data parallel example.

#### 2. hybrid parallel

Hybrid parallel includes data, tensor, and pipeline parallelism. In Colossal-AI, we support different types of tensor
parallelism (i.e. 1D, 2D, 2.5D and 3D). You can switch between different tensor parallelism by simply changing the configuration
in the `config.py`. You can follow the [GPT example](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt).
Detailed instructions can be found in its `README.md`.

#### 3. MoE parallel

We provided [an example of ViT-MoE](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/image/moe) to demonstrate
MoE parallelism. WideNet uses mixture of experts (MoE) to achieve better performance. More details can be found in
[Tutorial: Integrate Mixture-of-Experts Into Your Model](../advanced_tutorials/integrate_mixture_of_experts_into_your_model.md)

#### 4. sequence parallel

Sequence parallel is designed to tackle memory efficiency and sequence length limit problems in NLP tasks. We provided
[an example of BERT](https://github.com/hpcaitech/ColossalAI/tree/main/examples/tutorial/sequence_parallel) in
[ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI/tree/main/examples). You can follow the `README.md` to execute the code.

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 run_demo.py  -->

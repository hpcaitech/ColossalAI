# Shardformer

Author: [Baizhou Zhang](https://github.com/Fridge003)

**Prerequisite**
- [Paradigms of Parallelism](../concepts/paradigms_of_parallelism.md)
- [Booster API](../basics/booster_api.md)
- [Booster Plugins](../basics/booster_plugins.md)

**Example Code**
- [Tensor Parallelism with Shardformer](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer/examples)
- [Enabling Shardformer using HybridPrallelPlugin](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert)

**Related Paper**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)


## Introduction

When training large transformer models such as LLaMa-2 70B or OPT 175B, model parallelism methods that divide a huge model into smaller shards, including tensor parallelism or pipeline parallism, are essential so as to meet the limitation of GPU memory.
However, manually cutting model and rewriting its forward/backword logic could be difficult for users who are not familiar with distributed training.
Meanwhile, the Huggingface transformers library has gradually become users' first choice of model source, and most mainstream large models have been open-sourced in Huggingface transformers model library.

Out of this motivation, the ColossalAI team develops **Shardformer**, a feature that automatically does preparation of model parallelism (tensor parallelism/pipeline parallelism) for popular transformer models in HuggingFace.
This module aims to make parallelization hassle-free for users who are not from the system background.
Within a few lines of codes, users can turn a model into a state ready for distributed training.
Also, Shardformer contains various optimization tools for acceleration and memory saving during forward/backward pass.


## How Shardformer Works

Generally, Shardformer works through the following four kinds of *replacements*:

1. Replacing original PyTorch module (e.g. `nn.Linear`, `nn.Embedding`) with a crafted distributed module.
The distributed module keeps the same attributes as the original module but replaces the original parameters with distributed parameters.
Also, new `forward` methods will replace original ones so as to execute distributed computation, such as linear layers' split /gather operations executed under tensor parallelism.
Each distributed module implements its `from_native_module` static method to convert the PyTorch module to its corresponding distributed module.

2. Replacing attributes of original Huggingface Transformers layers with appropriate attributes for distributed training.
For example, when training LlaMa-2 with tensor parallel size as 2, the attribute `num_heads` of `LlamaDecoderLayer` (the number of attention heads in each layer) should be replaced with `model.config.num_attention_heads // 2`.

3. Replacing the `forward` methods implemented by original Huggingface
Transformers libraries with our customized `forward` methods.
This replacement is essential for pipeline paralellism, where a customiozed function is needed to pass intermediate hidden states between different pipeline stages.
Also, optimization methods such as flash attention or sequence parallel can be injected into the `forward` process through our customized `forward` method.

4. Replacing the whole copy of model parameters and optimizer states with incomplete ones controlled by current device (this is why it's called Shardformer).
By executing `ModelSharder.shard` method, current device will only keep the part of model parameters it's supposed to take care of.
To be specific, they should be the assigned parameter shards when using tensor parallelism, or the parameters belonging to current pipeline stage when using pipeline parallelism, or both of them.
All other parameters are released so as to liberate memory usage.
As a result, the optimizer will only compute the states corresponding to these part of parameters, causing the usage of memory to be further saved.

All of these replacements are implemented with manually written policies and forward functions.
If you want to delve deeper into the design of Shardformer or customize your own Shardformer policies, please refer to our [Shardformer development document](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md) and [pipeline parallelism design](https://github.com/hpcaitech/ColossalAI/discussions/4050) for more details.

## Usage

### Shardformer Configuration

The configuration of Shardformer is controlled by class `ShardConfig`:

{{ autodoc:colossalai.shardformer.ShardConfig }}

If you want to enable Apex Fused Layernorm, please install `apex`.
If you want to enable the usage of flash attention, please install `flash_attn`.
In addition, xFormers's `cutlass_op` can serve as a backup for flash attention.

### Enabling Shardformer

#### 1. Enabling Shardformer Through Booster (Recommended)

Enabling `Shardformer` through `Booster` initialized with `HybridParallelPlugin` is the recommended way to awaken the power of Shardformer.
The main reason is that pipeline parallelism cannot successfully work without the calling of `execute_pipeline` method of `Booster`. Besides, `HybridParallelPlugin` provides the capacity to combine the features of `Shardformer` with other useful features, such as mixed precision training or Zero.

More details about this usage can be found in chapter [Booster API](../basics/booster_api.md) and [Booster Plugins](../basics/booster_plugins.md).

[Here](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert) is an example on how to trigger `Shardformer` through `HybridParallelPlugin`. Please be aware that there's a difference in the way of doing forward and backward between the situation of using pipeline and not using pipeline.


#### 2. Enabling Shardformer Through Shardformer APIs (Not Recommended)

You can also use Shardformer through manually calling Shardformer APIs. However, this usage is not recommended since pipeline parallelism can't run without `Booster`.

[Here](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/convergence_benchmark.py)
is an example on how to trigger `Shardformer` through calling Shardformer APIs.


### Precautions

1. When enabling pipeline parallel, please don't do the forward/backward pass in the conventional way (`model(input)`, `loss.backward()`), which will cause unexpected errors. Rather, please do forward/backward pass through calling `booster.execute_pipeline` method.

2. When you use Shardformer to process classification models such as `GPT2ForSequenceClassification`, `ViTForImageClassification`, please ensure that the total number of labels should be integer multiple of tensor parallel size, otherwise Shardformer can't process the classifier layer correctly. A simple fix could be appending dummy labels in transformers config. This bug will be fixed in future version of Shardformer.

3. The case of training ChatGLM-2 6B is a little special: since Huggingface transformers doesn't officially support ChatGLM at present, please import the configuration/model classes through
    ```python
    from colossalai.shardformer.modeling.chatglm2_6b.configuration_chatglm import ChatGLMConfig
    from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel
    ```
    when training ChatGLM-2 with Shardformer, and initialize your model with these imported classes.


## Supporting Information

List of Huggingface transformers model families currently supported by Shardformer:
- LlaMa-1/LlaMa-2
- GPT2
- BERT
- OPT
- BLOOM
- T5
- ViT
- ChatGLM-2 6B
- Whisper

List of optimization tools currently supported by Shardformer:
- Flash Attention 2
- JIT Fused Operator
- xFormers
- Fused Layer Normalization
- Sequence Parallel
- Sequence Overlap

List of model families we plan to support in the near future:
- SAM
- Blip2
- RoBERTa
- ALBERT
- ERNIE
- GPT Neo
- GPT-J
- BEiT
- SwinTransformer V1/V2
- qwen

These lists will grow longer as more models and optimization tools emerge in the future. If you have any suggestions on the models/optimization we should support, please feel free to mention it in [Issues](https://github.com/hpcaitech/ColossalAI/issues) section of our project.

For more details about compatibility between each optimization tool and each supported model, please refer to chapter Roadmap in our [develop document](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md).


<!-- doc-test-command: echo  -->

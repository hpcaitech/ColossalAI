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

When training large transformer models such as LLaMa-2 70B or OPT 175B, model parallelism methods that divide a huge model into smaller shards, including tensor parallelism or pipeline parallism, are essential so as to meet the limitation of GPU memory. However, manually cutting model and rewriting its forward/backword logic could be difficult for users who are not familiar with distributed training. Meanwhile, the Huggingface transformers has gradually become users' first choice of model source, and most mainstream large models have been open-sourced in Huggingface model library.

Out of this motivation, the ColossalAI team develops **Shardformer**, a feature that automatically does preparation of model parallelism (tensor parallelism/pipeline parallelism) for popular transformer models in HuggingFace. This module aims to make parallelization hassle-free for users who are not from the system background. Within a few lines of codes, users can turn a large pretrained Huggingface model into a state ready for distributed training. Also, Shardformer can be configured to adopt some optimization tools for acceleration and memory saving during forward/backward pass.


## How Shardformer works




For more implementation details, please refer to our [develop document](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md).


## Usage




The case of training ChatGLM-2 6B is a little special: since Huggingface transformers doesn't officially support ChatGLM at present, please import the configuration/model classes through
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

These lists will grow longer as more models and optimization tools emerge in the future. If you have any suggestions on the models/optimization we should support, please mention it in [Issues](https://github.com/hpcaitech/ColossalAI/issues) section of our project.

For more details about compatibility between each optimization tool and each supported model, please refer to chapter Roadmap in our [develop document](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md).






<!-- doc-test-command: echo  -->

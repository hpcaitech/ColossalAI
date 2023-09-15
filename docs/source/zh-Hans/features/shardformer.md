# Shardformer

Author: [Baizhou Zhang](https://github.com/Fridge003)

**预备知识**
- [并行技术](../concepts/paradigms_of_parallelism.md)
- [Booster API](../basics/booster_api.md)
- [Booster 插件](../basics/booster_plugins.md)

**示例代码**
- [使用Shardformer进行张量并行训练](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer/examples)
- [通过HybridParallelPlugin使用Shardformer](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert)

**相关论文**
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)


## 简介

在训练LLaMa-2 70B或OPT 175B等大型Transformer模型时，为了满足GPU内存的限制，将大型模型划分为更小的分片的模型并行方法（包括张量并行以及流水线并行）是必不可少的。然而，对于不熟悉分布式训练的用户来说，手动剪切模型并重写其前向/反向逻辑可能很困难。与此同时，Huggingface transformers开源库正在逐渐成为用户模型来源的首选，大部分主流大型模型都已在Huggingface transformers模型库中开源。

出于这种动机，ColossalAI团队开发了**Shardformer**，该功能可以自动为HuggingFace中主流的Transformer模型进行封装，用于张量并行以及流水线并行的训练策略。如此一来，对系统了解不多的用户也可以轻松地在transformers模型上进行并行训练：只需几行代码，用户就可以将模型转变为并行训练的状态。此外，Shardformer也包括了多种优化工具，用于在前向/后向的传递过程中实现加速和节省内存。


## Shardformer的工作原理

通常来说，Shardformer通过以下四种“替换”进行工作：

1. 用我们设计的分布式模块替换原始的PyTorch模块（例如`nn.Linear`、`nn.Embedding`）。
分布式模块保持与原始模块相同的属性，但分布式模块会用新的参数替换原始模块的参数。新的前向函数将取代原来的前向函数，用于执行分布式计算，例如在张量并行下执行线性层的split/gather操作。每个分布式模块都应当实现其`from_native_module`静态方法，以将PyTorch模块转换为其相应的分布式模块。

2. 将原始Huggingface Transformers中间层的属性为适用于并行训练的属性。例如，当使用并行度为2的张量并行训练LlaMa-2时,`LlamaDecoderLayer`   的属性`num_heads`（每一层注意力头的数量）应替换为`model.config.num_attention_heads // 2`。

3. 将原来Huggingface transformers库实现的前向函数替换为我们定制的前向函数。前向函数的替换对于流水线并行性至关重要，因为流水线并行需要特殊的前向函数去在不同的流水线阶段之间传递中间的隐藏状态。此外，可以通过我们定制的前向函数将例如`flash attention`或序列并行的优化方法注入到前向的过程中。

4. 将完整的模型参数和优化器状态替换为只由当前设备控制的部分模型参数和优化器状态。通过执行`ModelSharder.shard`方法，当前设备仅会保留它应该处理的那部分模型参数。具体来说，这部分参数可以是使用张量并行时分配到当前机器的参数分片，或者使用流水线并行时当前流水线阶段的模型参数，或者兼而有之。除此之外的所有其他参数都被释放，用于节省内存的空间。
如此一来，优化器只会计算保留的部分参数对应的状态，从而进一步节省内存的使用。

所有这些替换都是通过手动编写的策略和前向函数来实现的。如果您想更深入地研究Shardformer的设计方案，或者定制您自己的Shardformer策略，请参考[Shardformer 开发者文档](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md)和[流水并行设计方案](https://github.com/hpcaitech/ColossalAI/discussions/4050)以获得更多细节。

## 用法

### Shardformer的参数配置

Shardformer的配置由类`ShardConfig`的参数控制：

{{ autodoc:colossalai.shardformer.ShardConfig }}

如果您想启用 Apex Fused Layernorm，请安装 `apex`。如果您想启用 flash attention，请安装 `flash_attn`。此外，xFormers 的 `cutlass_op` 可以作为Flash Attention的补充优化方式。

### 启动Shardformer

#### 1. 通过Booster启动Shardformer (推荐)

通过用`HybridParallelPlugin`初始化的`Booster`来启动`Shardformer`是最推荐的用法。其主要原因是如果不调用`Booster`的`execute_pipeline`方法，流水线并行就无法正常工作。此外，`HybridParallelPlugin`提供了将`Shardformer`的功能与其他功能（例如混合精度训练或Zero）相结合的能力。

更多关于这一用法的细节可以参考 [Booster API 文档](../basics/booster_api.md)以及[Booster 插件文档](../basics/booster_plugins.md)。[这里](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert)是一个通过`HybridParallelPlugin`启动`Shardformer`的示例。


#### 2. 通过Shardformer API启动Shardformer (不推荐)

您还可以通过手动调用Shardformer API的方式启动Shardformer。然而我们并不推荐这种用法，因为流水线并行在没有`Booster`的情况下无法正常运行。

[这里](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/convergence_benchmark.py)
是一个通过调用Shardformer的API启动`Shardformer`的示例。


### 注意事项

1. 当启用流水线并行时，请不要用常规方式（`model(input)`、`loss.backward()`）进行前向/后向传递，这样会导致未知的错误。这种情形下请通过调用`booster.execute_pipeline`方法来进行前向/后向传递。

2. 当使用Shardformer处理`GPT2ForSequenceClassification`、`ViTForImageClassification`等分类模型时，请确保labels的总数为张量并行度的整数倍，否则Shardformer无法正确地处理classifier层。一个简单的修复方法就是在transformers的config中添加虚拟的标签。这一bug将在 Shardformer的未来版本中修复。

3. 训练ChatGLM-2 6B的情况有点特殊：由于Huggingface Transformers 目前尚未正式支持ChatGLM。在使用Shardformer训练ChatGLM-2时，请通过以下方式导入config/model的类：
    ```python
    from colossalai.shardformer.modeling.chatglm2_6b.configuration_chatglm import ChatGLMConfig
    from colossalai.shardformer.modeling.chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMModel
    ```
    并且使用这些导入的类初始化模型。

## 支持信息

Shardformer目前支持的Huggingface Transformer模型:
- LlaMa-1/LlaMa-2
- GPT2
- BERT
- OPT
- BLOOM
- T5
- ViT
- ChatGLM-2 6B
- Whisper

Shardformer目前支持的优化工具:
- Flash Attention 2
- JIT Fused Operator
- xFormers
- Fused Layer Normalization
- Sequence Parallel
- Sequence Overlap

我们计划在不久后为Shardformer支持的模型:
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

随着未来更多模型和优化工具的出现，这些列表将会变得越来越长。如果您对我们应该支持的模型/优化工具有任何建议，欢迎在项目的[Issues](https://github.com/hpcaitech/ColossalAI/issues)板块参与讨论。

更多关于不同优化工具和模型之间兼容性的细节，请参考[Shardformer开发者文档](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md)中的Roadmap一节。

<!-- doc-test-command: echo  -->

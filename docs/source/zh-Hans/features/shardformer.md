# Shardformer

Author: [Baizhou Zhang](https://github.com/Fridge003), [Bin Jia](https://github.com/FoolPlayer)

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
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)


## 简介

在训练LLaMa-2 70B或OPT 175B等大型Transformer模型时，为了满足GPU内存的限制，将大型模型划分为更小的分片的模型并行方法（包括张量并行以及流水线并行）是必不可少的。然而，对于不熟悉分布式训练的用户来说，手动剪切模型并重写其前向/反向逻辑可能很困难。与此同时，Huggingface transformers开源库正在逐渐成为用户模型来源的首选，大部分主流大型模型都已在Huggingface transformers模型库中开源。

出于这种动机，ColossalAI团队开发了**Shardformer**，该功能可以自动为HuggingFace中主流的Transformer模型进行封装，用于张量并行以及流水线并行的训练策略。如此一来，对系统了解不多的用户也可以轻松地在transformers模型上进行并行训练：只需几行代码，用户就可以将模型转变为并行训练的状态。此外，Shardformer也包括了多种优化工具，用于在前向/后向的传递过程中实现加速和节省内存。

## 支持信息

模型/功能 兼容性矩阵：

<table>
  <tr>
    <th nowrap="nowrap">Model/Feature</th>
    <th nowrap="nowrap" title="Tensor Parallel">Tensor<br />Parallel</th>
    <th nowrap="nowrap" align="center" title="Pipeline Parallel">Pipeline<br />Parallel</th>
    <th nowrap="nowrap" align="center" title="Lazy Initialization">Lazy<br />Initialization</th>
    <th nowrap="nowrap" align="center" title="xFormers">xFormers</th>
    <th nowrap="nowrap" align="center" title="Flash Attention 2">Flash<br />Attention 2</th>
    <th nowrap="nowrap" align="center" title="JIT Fused Operators">JIT Fused<br />Operators</th>
    <th nowrap="nowrap" align="center" title="Fused LayerNorm">Fused<br />LayerNorm</th>
    <th nowrap="nowrap" align="center" title="Sequence Parallel">Sequence<br />Parallel</th>
    <th nowrap="nowrap" align="center" title="Sequence Overlap">Sequence<br />Overlap</th>
  </tr>
  <tr>
    <td nowrap="nowrap">Llama V1/V2</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">OPT</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
    <tr>
    <td nowrap="nowrap">BLOOM</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
  </tr>
  <tr>
    <td nowrap="nowrap">ChatGLM 2</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
  </tr>
  <tr>
    <td nowrap="nowrap">BERT</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
  </tr>
  <tr>
    <td nowrap="nowrap">GPT 2</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
  </tr>
  <tr>
    <td nowrap="nowrap">T5</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">ViT</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">Whisper</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">SAM</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">Blip2</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">Falcon</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td colspan="39"></td>
  </tr>
</table>

我们计划在不久后为Shardformer支持的模型:
- RoBERTa
- ALBERT
- ERNIE
- GPT Neo
- GPT-J
- BEiT
- SwinTransformer V1/V2
- qwen

随着未来更多模型和优化工具的出现，我们支持的模型/优化工具将会变得越来越多。如果您对我们应该支持的模型/优化工具有任何建议，欢迎在项目的[Issues](https://github.com/hpcaitech/ColossalAI/issues)板块参与讨论。

## 用法

### Shardformer的参数配置

Shardformer的配置由类`ShardConfig`的参数控制：

{{ autodoc:colossalai.shardformer.ShardConfig }}

如果您想启用 Apex Fused Layernorm，请安装 `apex`。如果您想启用 flash attention，请安装 `flash_attn`。此外，xFormers 的 `cutlass_op` 可以作为Flash Attention的补充优化方式。

### 启动Shardformer

#### 1. 通过Booster启动Shardformer (推荐)

通过用`HybridParallelPlugin`初始化的`Booster`来启动`Shardformer`是最推荐的用法。其主要原因是如果不调用`Booster`的`execute_pipeline`方法，流水线并行就无法正常工作。此外，`HybridParallelPlugin`提供了将`Shardformer`的功能与其他功能（例如混合精度训练或Zero）相结合的能力。

[这里](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert)是一个通过`HybridParallelPlugin`启动`Shardformer`的示例。
移动到示例的根目录下，执行命令：
```bash
torchrun --standalone --nproc_per_node 4  finetune.py --target_f1 0.86 --plugin "hybrid_parallel" --model_type "bert"
```
你便可以微调一个被`Shardformer`封装过的Bert模型，而封装的操作是由`HybridParallelPlugin`完成的。

接下来一起深入挖掘一下`finetune.py`里的代码：

在`main`函数中，混合并行的插件通过以下的代码创建
```python
...
elif args.plugin == "hybrid_parallel":
    # modify the param accordingly for finetuning test cases
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=2,
        num_microbatches=None,
        microbatch_size=1,
        enable_all_optimization=True,
        zero_stage=1,
        precision="fp16",
        initial_scale=1,
    )
```
在这里你可以通过设置不同的`tp_size`, `pp_size` 或 `zero_stage`来改变插件的配置。更多关于插件配置的信息可以在[Booster 插件文档](../basics/booster_plugins.md)中被找到。

当流水并行不被启用的时候，训练的流程和其他的插件是一样的 （先用Booster封装模型和优化器，再用正常的方式做前向和后向传递）。然而，当流水线并行被启用的时候，有几处不同于寻常情况的用法：

1. 在进行前向和后向之前，criterion函数（loss函数）需要被处理以满足流水线并行的传参要求:
    ```python
    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss
    ```

2. 在 `train_epoch` 函数中, dataloader 在进行流水线的前向后向操作之前需要被转换为 `Iterator` 类:
    ```python
    train_dataloader_iter = iter(train_dataloader)
    ```

3. 通过调用`Booster.execute_pipeline` 方法来执行前向和后向传递:
    ```python
    outputs = booster.execute_pipeline(
        train_dataloader_iter, model, _criterion, optimizer, return_loss=True
    )
    ```
    该方法会自动执行后向传递，所以在执行该方法后不需要再调用 `loss.backward()`方法。
    更多关于 `Booster.execute_pipeline` 的信息可以参考 [Booster API 文档](../basics/booster_api.md)。

#### 2. 通过Shardformer API启动Shardformer (不推荐)

您还可以通过手动调用Shardformer API的方式启动Shardformer。然而我们并不推荐这种用法，因为流水线并行在没有`Booster`的情况下无法正常运行。

[这里](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/convergence_benchmark.py)
是一个通过调用Shardformer的API启动`Shardformer`的示例。
在示例代码的`train`函数中，模型被以下的几行代码进行封装：
```python
...
if dist.get_world_size() > 1:
    tp_group = dist.new_group(backend="nccl")

    # First create configuration for Shardformer
    shard_config = ShardConfig(
        tensor_parallel_process_group=tp_group,
        enable_tensor_parallelism=True,
        enable_all_optimization=True
    )

    # Then create ShardFormer object with created config
    shard_former = ShardFormer(shard_config=shard_config)

    # Finally shard the model using ShardFormer.optimize method
    model, _ = shard_former.optimize(model)
...
```

### 注意事项

1. 当启用流水线并行时，请不要用常规方式（`model(input)`、`loss.backward()`）进行前向/后向传递，这样会导致未知的错误。这种情形下请通过调用`booster.execute_pipeline`方法来进行前向/后向传递。

2. 当使用Shardformer处理`GPT2ForSequenceClassification`、`ViTForImageClassification`等分类模型时，请确保labels的总数为张量并行度的整数倍，否则Shardformer无法正确地处理classifier层。一个简单的修复方法就是在transformers的config中添加虚拟的标签。这一bug将在 Shardformer的未来版本中修复。


## Shardformer的工作原理

### 设计思想

通常来说，Shardformer通过以下四种“替换”进行工作：

1. 用我们设计的分布式模块替换原始的PyTorch模块（例如`nn.Linear`、`nn.Embedding`）。
分布式模块保持与原始模块相同的属性，但分布式模块会用新的参数替换原始模块的参数。新的前向函数将取代原来的前向函数，用于执行分布式计算，例如在张量并行下执行线性层的split/gather操作。每个分布式模块都应当实现其`from_native_module`静态方法，以将PyTorch模块转换为其相应的分布式模块。

2. 将原始Huggingface Transformers中间层的属性为适用于并行训练的属性。例如，当使用并行度为2的张量并行训练LlaMa-2时,`LlamaDecoderLayer`   的属性`num_heads`（每一层注意力头的数量）应替换为`model.config.num_attention_heads // 2`。

3. 将原来Huggingface transformers库实现的前向函数替换为我们定制的前向函数。前向函数的替换对于流水线并行性至关重要，因为流水线并行需要特殊的前向函数去在不同的流水线阶段之间传递中间的隐藏状态。此外，可以通过我们定制的前向函数将例如`flash attention`或序列并行的优化方法注入到前向的过程中。

4. 将完整的模型参数和优化器状态替换为只由当前设备控制的部分模型参数和优化器状态。通过执行`ModelSharder.shard`方法，当前设备仅会保留它应该处理的那部分模型参数。具体来说，这部分参数可以是使用张量并行时分配到当前机器的参数分片，或者使用流水线并行时当前流水线阶段的模型参数，或者兼而有之。除此之外的所有其他参数都被释放，用于节省内存的空间。
如此一来，优化器只会计算保留的部分参数对应的状态，从而进一步节省内存的使用。

所有这些替换都是通过手动编写的策略和前向函数来实现的。如果您想更深入地研究Shardformer的设计方案，或者定制您自己的Shardformer策略，请参考[Shardformer 开发者文档](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md)和[流水并行设计方案](https://github.com/hpcaitech/ColossalAI/discussions/4050)以获得更多细节。

### 序列并行 Sequence Parallelism

序列并行是`Shardformer`支持的一种特殊的优化方法。在`Shardformer`中，序列并行与[此处](https://colossalai.org/docs/basics/configure_parallelization/#sequence-parallel)稍有不同，后者侧重于ring attention。在`Shardformer`中，序列并行仅与1D张量并行一起使用，以进一步减少计算中activation的内存占用。

1. 在普通的[1D张量并行](https://colossalai.org/docs/features/1D_tensor_parallel)中，有两个通信操作$g$和$\vec{g}$，$g$在反向传播中进行一次全局归约以获取来自所有设备的梯度，而$\vec{g}$在正向传播中进行一次All-Reduce以获取来自所有设备的输出。

2. 当使用序列并行时，$\vec{g}$需要在正向传播过程中进行All-Gather以获取序列维度上的输入，并在反向传播过程中进行Reduce-Scatter以分割梯度。$\vec{g}$需要进行Reduce-Scatter以将序列维度上的行线性层输出分割到所有设备上，并进行All-Gather以获取完整的梯度。

3. 使用NCCL的All-reduce实现采用了`Ring All-Reduce`方法，由一次Reduce-Scatter和一次All-Gather组成，两者的开销相等。因此，与序列并行和张量并行相比，它并不会引入额外的通信开销。

4. 需要注意的一点是，在张量并行的 `Column Linear` 层中进行序列并行时，梯度的反向计算过程中需要获取完整的输入。在前向传播过程中，仅保留沿序列维度分割的输入部分，张量的形状例如$(batch, sequence\_len/k, hidden\_states)$。因此，需要进行额外的全局收集操作以获取完整的输入进行梯度计算。但是，在实现中，可以将梯度计算与全局收集通信操作重叠，这不会引入额外的通信开销（对应`Shardformer`中的`enable_sequence_overlap`参数）。


<!-- doc-test-command: echo  -->

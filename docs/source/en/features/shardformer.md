# Shardformer

Author: [Baizhou Zhang](https://github.com/Fridge003), [Bin Jia](https://github.com/FoolPlayer)

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
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

## Introduction

When training large transformer models such as LLaMa-2 70B or OPT 175B, model parallelism methods that divide a huge model into smaller shards, including tensor parallelism or pipeline parallelism, are essential so as to meet the limitation of GPU memory.
However, manually cutting model and rewriting its forward/backword logic could be difficult for users who are not familiar with distributed training.
Meanwhile, the Huggingface transformers library has gradually become users' first choice of model source, and most mainstream large models have been open-sourced in Huggingface transformers model library.

Out of this motivation, the ColossalAI team develops **Shardformer**, a feature that automatically does preparation of model parallelism (tensor parallelism/pipeline parallelism) for popular transformer models in HuggingFace.
This module aims to make parallelization hassle-free for users who are not from the system background.
Within a few lines of codes, users can turn a model into a state ready for distributed training.
Also, Shardformer contains various optimization tools for acceleration and memory saving during forward/backward pass.

## Supporting Information

Model/Feature Compatibility Matrix:

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
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
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
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
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
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
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

List of model families we plan to support in the near future:
- RoBERTa
- ALBERT
- ERNIE
- GPT Neo
- GPT-J
- BEiT
- SwinTransformer V1/V2
- qwen

The support matrix will grow larger as more models and optimization tools emerge in the future. If you have any suggestions on the models/optimization we should support, please feel free to mention it in [Issues](https://github.com/hpcaitech/ColossalAI/issues) section of our project.

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

[Here](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/bert) is an example on how to trigger `Shardformer` through `HybridParallelPlugin`. Move to the root directory of this example, and execute
```bash
torchrun --standalone --nproc_per_node 4  finetune.py --target_f1 0.86 --plugin "hybrid_parallel" --model_type "bert"
```
Then you can start finetuning a bert model wrapped by `Shardformer`. The process of wrapping is operated by `HybridParallelPlugin`.

Let's delve into the code of `finetune.py`:

In the `main` function, the plugin is created through the following codes:
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
Here you can change the configuration of plugin by setting `tp_size`, `pp_size` or `zero_stage` to other values. More details about plugin configuration can be found in [Booster Plugins Doc](../basics/booster_plugins.md).

If pipeline parallel is not enabled, just do the training in the same way of other booster plugins(first boost with Booster, then do forward and backward through normal way).
However, if pipeline parallel is enabled, there are several usages different from other normal cases:

1. Before doing forward or backward, the criterion function (loss function) is processed to meet the argument demand of running pipeline:
    ```python
    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss
    ```

2. In `train_epoch` function, dataloader is converted into `Iterator` class before running pipeline:
    ```python
    train_dataloader_iter = iter(train_dataloader)
    ```

3. Do forward and backward passing through calling `Booster.execute_pipeline` method:
    ```python
    outputs = booster.execute_pipeline(
        train_dataloader_iter, model, _criterion, optimizer, return_loss=True
    )
    ```
    Backward passing has been completed by this method, so there is no need to call `loss.backward()` after executing this method.
    More details about `Booster.execute_pipeline` can be found in [Booster API Doc](../basics/booster_api.md).


#### 2. Enabling Shardformer Through Shardformer APIs (Not Recommended)

You can also use Shardformer through manually calling Shardformer APIs. However, this usage is not recommended since pipeline parallelism can't run without `Booster`.

[Here](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/examples/convergence_benchmark.py)
is an example on how to trigger `Shardformer` through calling Shardformer APIs. In the `train` function of example code, the model is wrapped by `Shardformer` through the following few codes:
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

### Precautions

1. When enabling pipeline parallel, please don't do the forward/backward pass in the conventional way (`model(input)`, `loss.backward()`), which will cause unexpected errors. Rather, please do forward/backward pass through calling `booster.execute_pipeline` method.

2. When you use Shardformer to process classification models such as `GPT2ForSequenceClassification`, `ViTForImageClassification`, please ensure that the total number of labels should be integer multiple of tensor parallel size, otherwise Shardformer can't process the classifier layer correctly. A simple fix could be appending dummy labels in transformers config. This bug will be fixed in future version of Shardformer.

## How Shardformer Works

### Main Idea

Generally, Shardformer works through the following four kinds of *replacements*:

1. Replacing original PyTorch module (e.g. `nn.Linear`, `nn.Embedding`) with a crafted distributed module.
The distributed module keeps the same attributes as the original module but replaces the original parameters with distributed parameters.
Also, new `forward` methods will replace original ones so as to execute distributed computation, such as linear layers' split /gather operations executed under tensor parallelism.
Each distributed module implements its `from_native_module` static method to convert the PyTorch module to its corresponding distributed module.

2. Replacing attributes of original Huggingface Transformers layers with appropriate attributes for distributed training.
For example, when training LlaMa-2 with tensor parallel size as 2, the attribute `num_heads` of `LlamaDecoderLayer` (the number of attention heads in each layer) should be replaced with `model.config.num_attention_heads // 2`.

3. Replacing the `forward` methods implemented by original Huggingface
Transformers libraries with our customized `forward` methods.
This replacement is essential for pipeline parallelism, where a customized function is needed to pass intermediate hidden states between different pipeline stages.
Also, optimization methods such as flash attention or sequence parallel can be injected into the `forward` process through our customized `forward` method.

4. Replacing the whole copy of model parameters and optimizer states with incomplete ones controlled by current device (this is why it's called Shardformer).
By executing `ModelSharder.shard` method, current device will only keep the part of model parameters it's supposed to take care of.
To be specific, they should be the assigned parameter shards when using tensor parallelism, or the parameters belonging to current pipeline stage when using pipeline parallelism, or both of them.
All other parameters are released so as to liberate memory usage.
As a result, the optimizer will only compute the states corresponding to these part of parameters, causing the usage of memory to be further saved.

All of these replacements are implemented with manually written policies and forward functions.
If you want to delve deeper into the design of Shardformer or customize your own Shardformer policies, please refer to our [Shardformer development document](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/README.md) and [pipeline parallelism design](https://github.com/hpcaitech/ColossalAI/discussions/4050) for more details.

### Sequence Parallelism

Sequence parallelism is a special optimization method supported by `Shardformer`. Sequence parallelism in `Shardformer` is a little different from [this one](https://colossalai.org/docs/basics/configure_parallelization/#sequence-parallel) which focuses on ring attention. In `Shardformer`, sequence parallelism is only used along with 1D tensor parallelism to further reduce memory occupation of activation tensors during computation.

1. In normal [1D tensor parallel](https://colossalai.org/docs/features/1D_tensor_parallel), there are 2 communication operations, $g$ and $\vec{g}$, $g$ will do one time All-Reduce in backward to get all gradients from all the devices and $\vec{g}$ will do one time All-Reduce in forward to get whole outputs from all the devices.

2. When using sequence parallelism, $\vec{g}$ needs to do All-Gather to gather the inputs along sequence dimension during forward, and Reduce-Scatter to split the gradient during backward. $\vec{g}$ needs to do Reduce-Scatter to split the output of `Row Linear` layer of tensor parallel to all devices along sequence dimension, and All-Gather to get the whole gradient during backward.

3. NCCL's implementation of All-Reduce adopts the `Ring All-Reduce` approach, which consists of a Reduce-Scatter operation and an All-Gather operation with equal costs. Therefore, compared with sequence parallelism and tensor parallelism, it does not introduce additional communication overhead.

4. One important thing to note is that when using sequence parallelism along with `Column Linear` module of tensor parallelism, the complete input needs to be obtained during the backward computation of gradients. During the forward pass, only the portion of the input that is split along the sequence dimension is retained, in the shape of $(batch, sequence_len/k, hidden_states)$. Therefore, an additional All-Gather operation is required to obtain the complete input for gradient computation. However, it is possible to overlap the gradient computation with the All-Gather communication operation in our implementation, which would not introduce additional communication overhead (corresponding to the `enable_sequence_overlap` parameter in `Shardformer`).


<!-- doc-test-command: echo  -->

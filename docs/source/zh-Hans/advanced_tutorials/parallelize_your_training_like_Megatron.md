# 使用ColoTensor让串行程序像Megatron-LM一样并行

Author: [Haichen Huang](https://github.com/1SAA) and [Jiarui Fang](https://github.com/feifeibear)

**Prerequisite:**
- [ColoTensor Concepts](../basics/colotensor_concept.md)

## 介绍

在新版本中，我们引入了ColoTensor。ColoTensor为用户使用并行训练提供了极大的便利，使得用户可以在原本的串行代码上，通过较小的修改将训练改为并行。在本教程中，我们将说明如何修改训练模型以自动使代码采取像 Megatron-LM 一样的方式并行训练。我们以 HuggingFace 提供的 GPT-2 模型为例，并提供一种方式让你可以在单个GPU上预训练GPT-2模型。

Megatron-LM 提供了一个具有影响力的并行化范式，这个范式主要应用于Transformer大模型的训练。然而，为了大规模训练 Transformer 语言大模型，用户必须使用Megatron-LM提供的特殊模块来构建他们的模型。这给用户带来了一些困难的工作，例如从预先训练的模型中加载权重，或是构建自己的并行训练模型。为了减轻用户的麻烦，我们提供 ColoTensor 类，以完成自动启用张量模型并行。

## 定义模型和损失函数

首先，我们直接调用 HuggingFace 库中的 GPTModel 和 GPTLoss。

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=50257, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

## 对GPT-2的简短回顾

现在，我们回顾一下 GPT-2 模型的结构。每个 GPT-2 模型都可以表示为一个 DAG。如下图所示，每个圆圈代表一个算子，每个方块代表一个权重。每个箭头表示输入数据的流向，而箭头旁边的符号表示输入数据的形状。

然后，让我们深入了解一下这个 GPT-2 模型。它由三部分组成，分别是**嵌入模块**、**转换器层**和**分类头**。

嵌入模块包含两个权重，符号嵌入权重和位置嵌入权重。在嵌入模块的前向操作之后，原始输入数据的所有序列中的每个单词都会被嵌入到隐藏状态。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/omfkIEN6ui5jcL3.png"/>
<figcaption>嵌入模块</figcaption>
</figure>

每个转换器层包含两个块。自注意操作在第一个块中调用，同时一个双层感知器位于第二个块中。

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/LAVzDlpRcj4dYeb.png"/>
<figcaption>转换器层</figcaption>
</figure>

最后，分类头只是一个不加偏差的线性模块，里面只有一个线性权重。

## 应用ColoTensor

两个步骤使您的串行代码采取 Megatron-LM 张量并行风格。
1. 在ColoInitContext的上下文中初始化模型。
2. 为每个参数设置 ColoTensorSpec。

### 使用 ColoInitContext 初始化

我们应该在 ColoInitContext 中构建模型。在该种上下文中，任何初始化的参数都将转换为 ColoParameter 并自动移动到相应的设备上。

```python
from colossalai.utils.model.colo_init_context import ColoInitContext

with ColoInitContext(device=torch.device('cpu')):
    model = GPTLMModel()
```

### 为每个参数设置 ColoTensorSpec

模型创建完成后，我们通过ProcessGroup建立分布式环境。这里，我们将张量并行度指定为所有GPU的数量，即数据并行度为一。

```python
import torch.distributed as dist
from colossalai.tensor import ProcessGroup

pg = ProcessGroup(tp_degree=dist.get_world_size())
```

现在，我们需要一些辅助函数为下一步做准备。我们定义了两个函数来切分参数。Megatron-LM张量并行需要沿参数的第一维或最后一维切分参数张量。

```python
from colossalai.tensor import ShardSpec, ComputeSpec, ComputePattern, ColoParameter, ProcessGroup

def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    if param.process_group.tp_world_size() == 1:
        param.set_process_group(pg)
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)
```

然后我们使模型采用张量并行。根据 Megatron 中使用的张量并行，应该沿着张量的最后一个维度进行切片，包括符号嵌入的权重，位置嵌入的权重，自注意力块中的所有线性权重和偏差，以及每个双层感知器中的第一个线性权重和偏差。且需要沿第一个维度切分双层感知器中的第二个线性权重。

```python
for mn, module in model.named_modules():
    for pn, param in module.named_parameters(recurse=False):
        # set process group for all parameters
        param.set_process_group(pg)

        if 'mlp.c_fc' in mn:
            if 'weight' in pn or 'bias' in pn:
                split_param_col_tp1d(param, pg)  # column slice
                # keep the shape of the output from c_fc
                param.compute_spec.set_output_replicate(False)
        elif 'mlp.c_proj' in mn:
            if 'weight' in pn:
                split_param_row_tp1d(param, pg)  # row slice
        elif 'wte' in mn or 'wpe' in mn:
            split_param_col_tp1d(param, pg)  # column slice
        elif 'c_attn' in mn or 'c_proj' in mn:
            split_param_col_tp1d(param, pg)  # column slice
```

修改后的模型如下图所示。

嵌入模块:

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/Yu2xzXEabHV7pwe.png"/>
<figcaption>修改后的嵌入模块</figcaption>
</figure>

转换器层:

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/4HWsA2xz51IhPFO.png"/>
<figcaption>修改后的转换器层</figcaption>
</figure>

一旦用户指定了每个参数的在并行中的分布模式，ColoTensor 就能够推断出所有算子的计算模式，包括矩阵乘法、线性函数、torch.nn.functional 中的其他逐元素函数，以及其他的一些常用函数。这样，用户可以像往常一样训练他们的模型。

在我们最新示例中还定义了一个Gemini + ZeRO DDP 的模型从而减小开销，提升效率。这一部分的详细内容可以参考[ZeRO](../features/zero_with_chunk.md)，你可以将这两部分内容结合起来看从而理解我们整个训练流程：

```python
def gemini_zero_dpp(model: torch.nn.Module, pg: ProcessGroup, placement_policy: str = "auto"):
    from colossalai.nn.parallel import GeminiDDP
    model = GeminiDDP(model,
                        device=get_current_device(),
                        placement_policy=placement_policy,
                        pin_memory=True,
                        search_range_m=32)
    return model
```

## 在单个GPU上预训练GPT-2

我们做的上述优化让我们可以在单GPU上训练GPT-2模型，只需要将`run.sh`中设置参数`GPUNUM`=1，再运行文件时就可以在单个GPU上完成模型的训练。

GPT-2 示例在[Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt). 获得。


<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 parallelize_your_training_like_Megatron.py  -->

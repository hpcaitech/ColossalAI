# Parallelize Your Training like Megatron-LM via ColoTensor

Author: [Haichen Huang](https://github.com/1SAA) and [Jiarui Fang](https://github.com/feifeibear)

**Prerequisite:**
- [ColoTensor Concepts](../basics/colotensor_concept.md)

## Introduction

Thanks to the convenience given by ColoTensor, users can apply parallelism with the least edition to their serial code.
In this tutorial, we will illustrate how to modify the training model to automatically adapt the code to parallel training like Megatron-LM.
We take the GPT-2 model offered by HuggingFace as an example and provide a way for you to pre-train the GPT-2 model on a single GPU.

Megatron-LM provided a profound paradigm to parallelize large transformer language models.
However, in order to train large transformer language models at scale, users have to build their models with those modules provided by Megatron.
It imposes several difficult jobs on users, such as loading the weights from the pre-trained models and constructing the parallelized models.
To mitigate users' trouble, we offer ColoTensor to enable the tensor model parallelism automatically.

## Definitions of the model and the loss function

First we use the GPTModel and GPTLoss directly from the HuggingFace library.

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

## Brief Review of GPT-2

Now, we recall the structure of each GPT-2 model.
Every GPT-2 model can be represented as a DAG.
As shown in the below pictures, each circle represents an operator and each square represents a weight.
An arrow indicates the flow of the input data, and the notation alongside the arrow demonstrates the shape of the input data.

Then, let's take an insight into this GPT-2 model. It consists of three parts.
They are the **embedding module**, **transformer layers**, and the **classification head**.

The embedding module contains two weights, token embedding weight and position embedding weight.
After the forward operation of the embedding module, each word in all sequences of the raw input data will be embedded into a hidden state.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/omfkIEN6ui5jcL3.png"/>
<figcaption>The embedding module</figcaption>
</figure>

Each transformer layer contains two blocks. The self-attention operation is called in the first block and a two-layer perception is located in the second block.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/LAVzDlpRcj4dYeb.png"/>
<figcaption>The transformer layer</figcaption>
</figure>

In the end, the classification head is just a linear module without bias, which only has a weight inside.

## Applied with ColoTensor

Two steps make your serial code adapted to Megatron-LM tensor parallel style.
1. Initialize the model in the context of ColoInitContext.
2. Setting ColoTensorSpec for each parameter.

### Initialize with ColoInitContext

We should build the model in the ColoInitContext.
In this context, any parameter initialized would be transformed to ColoParameter and moved to the corresponded device automatically.

```python
from colossalai.utils.model.colo_init_context import ColoInitContext

with ColoInitContext(device=torch.device('cpu')):
    model = GPTLMModel()
```

### Setting ColoTensorSpec for each parameter

After the creation of the model, we establish the distributed environment through ProcessGroup.
Here, we specify the degree of the tensor parallelism as the same as the number of all GPUs, which means the degree of data parallelism is 1.

```python
import torch.distributed as dist
from colossalai.tensor import ProcessGroup

pg = ProcessGroup(tp_degree=dist.get_world_size())
```

Now, some auxiliary functions are necessary for the next step. We define two functions to split a parameter.
Megatron-LM-like tensor parallelism requires splitting a parameter tensor along its first dimension or its last dimension.

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

Then we adapt the model to the tensor parallelism.
According to the tensor parallelism applied in Megatron, it is supposed to shard along the last dimension of tensors, including the weights of token embedding, position embedding, all linear weights and biases in self-attention blocks, the first weight linear and bias in each MLP.
And it shards the second linear weight along its first dimension.

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

The modified model is illustrated below.

The embedding module:

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/Yu2xzXEabHV7pwe.png"/>
<figcaption>The modified embedding module</figcaption>
</figure>

The transformer layers:

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/08/17/4HWsA2xz51IhPFO.png"/>
<figcaption>The modified transformer layer</figcaption>
</figure>

Once users have specified the distributed pattern of each parameter, ColoTensor is capable of inferring the computation patterns of all operators, including matrix multiplication, the linear function, other elementwise functions in torch.nn.functional, etc.
In this way, users can train their models as usual.

In our latest example, a Gemini + ZeRO DDP model is also defined to reduce overhead and improve efficiency.For the details of this part, please refer to [ZeRO](../features/zero_with_chunk.md). You can combine these two parts to understand our entire training process:

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

## Pretrain GPT-2 On Single GPU

The above optimization we made allows us to pretrain the GPT-2 model on a single GPU. We only need to set the parameter `GPUNUM`=1 in `run.sh`, and then we can complete the model training on a single GPU when running the file.

The GPT-2 example is accessible at [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt).

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 parallelize_your_training_like_Megatron.py  -->

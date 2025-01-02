# ZeroBubble Pipeline Parallelism
Author: [Junwen Duan](https://github.com/duanjunwen), [Hongxin Liu](https://github.com/ver217)

**Related Paper**
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

## Introduction
ZeroBubble (V Schedule):
Crucially, splitting B into two stages (also known as an activation gradient and a weight gradient) and a scheme like 1F1B1W can further reduce the bubble compared to the 1F1B scheme in earlier work.

## Hands-On Practice
We now demonstrate how to use ZeroBubble with booster API with 4 GPUs.

### step 1. Import libraries
```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import HybridParallelPlugin
from colossalai.pipeline.schedule.zero_bubble_pp import ZeroBubbleVPipeScheduler
```

### step 2. Initialize Distributed Environment and Parallism Group
```python
colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
```

### step 3. Initialize Module, Optimizer, and Pipeline Schedule
Build our model and Optimizer. We created a Llama with 8 Decoder-Layer. Then, inite the PipelineGraph and Pipeline schedule by get_v_schedule() function.
```python
# Global Param
NUM_BATCH = 8
NUM_TOK_PER_BATCH = 4
NUM_LAYERS = 8
HIDDEN_SIZE_PER_HEAD = 4
NUM_HEADS = 4
# Init Llama from huggingface
configuration = LlamaConfig(
    hidden_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS,
    intermediate_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS * 2,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_HEADS,
    attn_implementation="flash_attention_2",
)
model = LlamaModel(configuration).cuda()
optimizer = torch.optim.Adam(torch_model.parameters(), lr=1)
```
### step 4. Initialize Module, Optimizer, and Pipeline Schedul
Then, we need to create the PipelineGraph and PipelineSchedule using the get_v_schedule() function. We need to initialise the PipelineGraph with the following parameters.
x_cost represents the runtime consumed by operation x of each model chunk.
x_mem represents the amount of memory consumed by the operation x of each model chunk.
These parameters are estimated and filled in before the pipeline starts. In fact, better results can be obtained based on the runtime and memory cost during the real computation of the model.
In the following example, we assume that the computation times for the model's forward, reverse B, and reverse W are 1, 1, 1, respectively, and the p2p communication time is 1.
```python
# Init schedule
h, a, s = config.hidden_size, config.num_attention_heads, 1024
mem_f = 34 * h + 5 * a * s
mem_w = -32 * h
mem_b = -mem_w - mem_f
graph = PipelineGraph(
    n_stage=pp_size,
    n_micro=num_microbatches,
    f_cost=1,
    b_cost=1,
    w_cost=1,
    c_cost=1,
    f_mem=mem_f,
    b_mem=mem_b,
    w_mem=mem_w,
)
zbv_schedule = graph.get_v_schedule()
```

### step 5.Init Booster
Pass pp_style="zbv" when initialising the Plugin to use the ZeroBubble Pipeline.
```python
plugin = HybridParallelPlugin(
    pp_size=4,
    num_microbatches=4,
    tp_size=1,
    sp_size=1,
    zero_stage=1,
    initial_scale=1,
    find_unused_parameters=True,
    pp_style="zbv",
    scheduler_nodes=zbv_schedule,
    num_model_chunks=2,
)

dp_size = plugin.dp_size
booster = Booster(plugin=plugin)
```

### step 6.Train Your Model
```python
steps = 10
for step in range(steps):
    input_embeddings = torch.rand(
        NUM_BATCH, NUM_TOK_PER_BATCH, HIDDEN_SIZE_PER_HEAD * NUM_HEADS, requires_grad=True
    ).cuda()
    dist.all_reduce(
        input_embeddings, group=plugin.pp_group
    )
    data_iter = iter([{"inputs_embeds": input_embeddings}])
    output = booster.execute_pipeline(
        data_iter,
        model,
        lambda x, y: x.last_hidden_state.mean(),
        optimizer,
        return_loss=True,
        return_outputs=True,
    )
    optimizer.step()
    optimizer.zero_grad()
```

## Advanced Practice
In ColossalAI, you can get better training performance by using MetaCache and HybridParallel with ZeroBubble.
### 1.Use MetaCache with ZeroBubble
Pass "enable_metadata_cache=True" when initialising the Plugin to use the Meta Cache with ZeroBubble Pipeline.
```python
plugin = HybridParallelPlugin(
    pp_size=2,
    num_microbatches=4,
    tp_size=2,
    sp_size=2,
    zero_stage=1,
    initial_scale=1,
    enable_metadata_cache=True,
    find_unused_parameters=True,
    pp_style="zbv",
    scheduler_nodes=zbv_schedule,
    num_model_chunks=2,
)
```

### 2.HybridParallel with ZeroBubble
Pass pp_size, tp_size, sp_size when initialising the Plugin to use the HybridParallel with ZeroBubble Pipeline.
```python
plugin = HybridParallelPlugin(
    pp_size=2,
    num_microbatches=2,
    tp_size=2,
    sp_size=2,
    zero_stage=1,
    initial_scale=1,
    find_unused_parameters=True,
    pp_style="zbv",
    scheduler_nodes=zbv_schedule,
    num_model_chunks=2,
)
```
Performance Benchmark
<table>
  <tr>
    <th nowrap="nowrap">HybridParallel Strategy</th>
    <th nowrap="nowrap" align="center">Pipeline Parallel</th>
    <th nowrap="nowrap" align="center">Sequence Parallel + Pipeline Parallel</th>
    <th nowrap="nowrap" align="center">Data Parallel + Pipeline Parallel</th>
  </tr>
<tr>
    <td nowrap="nowrap" align="center" title="1F1B">With 1F1B</td>
    <td nowrap="nowrap" align="center">15.27 samples/sec</td>
    <td nowrap="nowrap" align="center">17.22 samples/sec</td>
    <td nowrap="nowrap" align="center">14.06 samples/sec</td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="center" title="Zero Bubble">With Zero Bubble</td>
    <td nowrap="nowrap" align="center">17.36 samples/sec</td>
    <td nowrap="nowrap" align="center">18.38 samples/sec</td>
    <td nowrap="nowrap" align="center">14.44 samples/sec</td>
  </tr>
  <tr>
    <td colspan="39"></td>
  </tr>
</table>

### 3.Fine-tuning Scheduler parameters

```python
```
## Model compatibility
<table>
  <tr>
    <th nowrap="nowrap">Shardformer/Model</th>
    <th nowrap="nowrap" align="center">Bert</th>
    <th nowrap="nowrap" align="center">Blip2</th>
    <th nowrap="nowrap" align="center">Bloom</th>
    <th nowrap="nowrap" align="center">Chatglm2</th>
    <th nowrap="nowrap" align="center">Command</th>
    <th nowrap="nowrap" align="center">Deepseek</th>
    <th nowrap="nowrap" align="center">Falcon</th>
    <th nowrap="nowrap" align="center">GPT2</th>
    <th nowrap="nowrap" align="center">Gptj</th>
    <th nowrap="nowrap" align="center">Llama</th>
    <th nowrap="nowrap" align="center">Mistral</th>
    <th nowrap="nowrap" align="center">Opt</th>
    <th nowrap="nowrap" align="center">Qwen2</th>
    <th nowrap="nowrap" align="center">Sam</th>
    <th nowrap="nowrap" align="center">T5</th>
    <th nowrap="nowrap" align="center">Vit</th>
    <th nowrap="nowrap" align="center">Whisper</th>
  </tr>
  <tr>
    <td nowrap="nowrap" align="center" title="ZeroBubble">ZeroBubble</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
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
    <td colspan="39"></td>
  </tr>
</table>

## API Reference
{{ autodoc:colossalai.pipeline.schedule.zero_bubble_pp.ZeroBubbleVPipeScheduler }}

<!-- doc-test-command: torchrun --standalone --nproc_per_node=4 zerobubble_pipeline_parallelism.py  -->

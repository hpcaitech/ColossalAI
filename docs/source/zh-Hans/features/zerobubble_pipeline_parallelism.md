# 零气泡流水线并行
作者: [Junwen Duan](https://github.com/duanjunwen), [Hongxin Liu](https://github.com/ver217)

**相关论文**
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

## 介绍
零气泡（V Schedule）：
与早期工作中的1F1B方案相比，零气泡流水线并行将B分成两个阶段（也称为激活梯度和权重梯度），形如1F1B1W这样的方案可以进一步减少气泡。

## 使用
我们将演示如何在 4 个 GPU 上使用带有 booster API 的 ZeroBubble

### step 1. 引用仓库
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

### step 2. 初始化分布式环境
```python
colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
```

### step 3. 初始化模型优化器
建立我们的模型和优化器 我们创建了一个带有8层Decoder-Layer的 Llama。然后，使用get_v_schedule()函数创建PipelineGraph和Pipeline schedule。

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
### step 4.初始化流水线Schedule
然后，我们需要使用 get_v_schedule() 函数创建 PipelineGraph 和 PipelineSchedule。我们需要用以下参数初始化 PipelineGraph。
x_cost 表示每个模型块的操作 x 所消耗的运行时间。
x_mem 表示每个模型块的操作 x 所消耗的内存量。
这些参数都是在流水线启动前估算并填入的。事实上，在模型的实际计算过程中，根据运行时间和内存成本可以获得更好的结果。
在下面的例子中，我们假设模型的正向、反向 B 和反向 W 的计算时间分别为 1、1、1，p2p 通信时间为 1。
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

### step 5.初始化Booster
在初始化Plugin时输入pp_style="zbv"，以使用ZeroBubble流水线并行。
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

### step 6.训练模型
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

## 进阶使用技巧
在 ColossalAI 中，通过使用MetaCache和混合并行的ZeroBubble，可以获得更好的训练性能。

### 1.在ZeroBubble中使用元数据缓存
在初始化Plugin时输入 "enable_metadata_cache=True"，以便在ZeroBubble管道中使用元数据缓存。
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

### 2.同时使用ZeroBubble和混合并行
在初始化插件时传递 pp_size, tp_size, sp_size, 以便使用零气泡混合并行管道（HybridParallel with ZeroBubble Pipeline）。
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
性能指标
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

## 模型兼容性
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
    <td nowrap="nowrap" align="center">✔️</td>
  </tr>
  <tr>
    <td colspan="39"></td>
  </tr>
</table>

## API 参考
{{ autodoc:colossalai.pipeline.schedule.zero_bubble_pp.ZeroBubbleVPipeScheduler }}

<!-- doc-test-command: torchrun --standalone --nproc_per_node=4 zerobubble_pipeline_parallelism.py  -->

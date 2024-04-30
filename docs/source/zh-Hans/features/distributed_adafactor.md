# 分布式 Adafactor

作者:

**相关论文**
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## 简介

分布式 Adafactor 是一种支持混合优化的优化器，包括 1D 张量并行和 ZerO。它通过合理的任务并行化充分利用了计算资源，提高了训练效率和速度，并减少了存储压力。它应用广泛，目前支持一系列基于 Transformer 的模型，详见 [tests.kit.model_zoo](https://github.com/hpcaitech/ColossalAI/tree/main/tests/kit/model_zoo).

## API接口

{{ autodoc:colossalai.nn.optimizer.distributed_adafactor.DistributedAdaFactor }}

## 实例演示
现在我们演示如何使用 Booster API 启动分布式 Adafactor。
### 步骤 1. 导入相关库

```python
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaModel, LlamaConfig

from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.optimizer.distributed_adafactor import DistributedAdaFactor
from colossal_llama2.dataset.loader import load_tokenized_dataset
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
```

### 步骤 2. 初始化分布式环境和参数
然后，我们需要初始化分布式环境。为了演示的目的，我们使用了 `colossalai.launch`。您可以参考 [Launch Colossal-AI](../basics/launch_colossalai.md) 获得其他的初始化方法。这里, 我们使用 "ProcessGroupMesh"来创建张量并行组和数据并行组。

```python
# Distributed Enviroment
config = {}
colossalai.launch(config=config, rank=rank, world_size=world_size,host="localhost", port=port, backend="nccl")
```

### 步骤 3.初始化模块和优化器
Build our model. We created an MLP using two Linear Layer.

```python
# Init Llama from huggingface
configuration = LlamaConfig()
model = LlamaModel(configuration)
dataset = load_tokenized_dataset(dataset_paths=args.dataset, mode="train")
dataloader = plugin.prepare_dataloader(dataset, batch_size=8)
criterion = lambda x: x.mean()
dist_optim = DistributedAdaFactor(model.parameters())

```

### 步骤 4.初始化Booster

```python
plugin = LowLevelZeroPlugin()
booster = Booster(plugin=plugin)
model, dist_optim, criterion, dataloader, _ = booster.boost(model, dist_optim, criterion, dataloader)
```
### 步骤 5.训练模型
```python
for epoch in range(max_epochs):
    for input_ids, attention_mask in dataloader:
        outputs = model(input_ids.cuda(), attention_mask.cuda())
        loss = criterion(outputs.logits, input_ids)
        booster.backward(loss, optimizer)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

## 支持信息
模型/功能兼容性矩阵:
<table>
  <tr>
    <th nowrap="nowrap">Model/Feature</th>
    <th nowrap="nowrap" title="Transformers Bert">Transformers<br />Bert</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Pretraining">Transformers Bert<br />For Pretraining</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert Lm Head Model">Transformers Bert<br />Lm Head Model</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Masked Lm">Transformers Bert<br />For Masked Lm</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Sequence Classification">Transformers Bert<br />For Sequence Classification</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Token Classification">Transformers Bert<br />For Token Classification</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Next Sentence">Transformers Bert<br />For Next Sentence</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Multiple-choice Question">Transformers Bert<br />For Multiple-choice Question</th>
    <th nowrap="nowrap" align="center" title="Transformers Bert For Question Answering">Transformers Bert<br />For Question Answering</th>
  </tr>
  <tr>
    <td nowrap="nowrap">Hybrid Parallel<br />Plugin</td>
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
    <td nowrap="nowrap">Low Level Zero<br />Plugin</td>
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
    <td nowrap="nowrap">Torch DDP<br />Plugin</td>
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
    <td nowrap="nowrap">Gemini<br />Plugin</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap">Moe Hybrid<br />Plugin</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td colspan="39"></td>
  </tr>
</table>

<!-- doc-test-command: python -m pytest -rP ./tests/test_optimizer/test_dist_adafactor.py  -->

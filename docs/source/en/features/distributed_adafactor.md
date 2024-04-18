# Distributed Adafactor

Author:

**Related Paper**
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

## Introduction

Distributed Adafactor is an optimiser that supports hybrid optimisation, including 1D tensor parallelism as well as ZerO. It makes full use of computational resources through reasonable task parallelism, improves training efficiency and speed, and reduces space pressure on single card storage. It has a wide range of applications and currently supports a range of Transformer based models, see [tests.kit.model_zoo](https://github.com/hpcaitech/ColossalAI/tree/main/tests/kit/model_zoo) for details.

## API Reference

{{ autodoc:colossalai.nn.optimizer.distributed_adafactor.DistributedAdaFactor }}

## Hands-On Practice
We now demonstrate how to start Distributed Adafactor with booster API.
### step 1. Import libraries

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

### step 2. Initialize Distributed Environment and Parallism Group
We then need to initialize distributed environment. For demo purpose, we uses `colossalai.launch`. You can refer to [Launch Colossal-AI](../basics/launch_colossalai.md)
for other initialization methods. We use `ProcessGroupMesh` to create tensor parallelism group and data parallelism group.

```python
# Distributed Enviroment
config = {}
colossalai.launch(config=config, rank=rank, world_size=world_size,host="localhost", port=port, backend="nccl")
```

### step 3. Initialize Module and Optimizer
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

### step 4.Init Booster

```python
plugin = LowLevelZeroPlugin()
booster = Booster(plugin=plugin)
model, dist_optim, criterion, dataloader, _ = booster.boost(model, dist_optim, criterion, dataloader)
```
### step 5.Train Your Model
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

## Supporting Information
Model/Feature Compatibility Matrix:
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

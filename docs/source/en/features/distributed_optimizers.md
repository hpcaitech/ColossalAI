# Distributed Optimizers

Author: [Wenxuan Tan](https://github.com/Edenzzzz), [Junwen Duan](https://github.com/duanjunwen), [Renjie Mao](https://github.com/chongqichuizi875)

**Related Paper**
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
- [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/pdf/1904.00962)

## Introduction
Apart from the widely adopted Adam and SGD, many modern optimizers require layer-wise statistics to update parameters, and thus aren't directly applicable to settings where model layers are sharded across multiple devices. We provide optimized distributed implementations with minimal extra communications, and seamless integrations with Tensor Parallel, DDP and ZeRO plugins, which automatically uses distributed optimizers with 0 code change.

## Optimizers
Adafactor is a first-order Adam variant using Non-negative Matrix Factorization(NMF) to reduce memory footprint. CAME improves by introducting a confidence matrix to correct NMF. GaLore further reduces memory by projecting gradients into a low-rank space and 8-bit block-wise quantization. Lamb allows huge batch sizes without lossing accuracy via layer-wise adaptive update bounded by the inverse of its Lipschiz constant.


## Hands-On Practice
We now demonstrate how to use Distributed Adafactor with booster API combining Tensor Parallel and ZeRO 2 with 4 GPUs. **Note that even if you're not aware of distributed optimizers, the plugins automatically casts yours to the distributed version for convenience.**
### step 1. Import libraries

```python
from transformers import LlamaModel, LlamaConfig
from colossalai.nn.optimizer.distributed_adafactor import DistributedAdaFactor
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
import colossalai
import torch
```

### step 2. Initialize Distributed Environment and Parallism Group
We need to initialize distributed environment. For demo purpose, we use `colossal run --nproc_per_node 4`. You can refer to [Launch Colossal-AI](../basics/launch_colossalai.md)

```python
colossalai.launch_from_torch()
```

### step 3. Initialize Module and Optimizer
Build our model. We created an MLP using two Linear Layer.

```python
# Init Llama from huggingface
configuration = LlamaConfig()
model = LlamaModel(configuration).cuda()
criterion = lambda x: x.mean()
dist_optim = DistributedAdaFactor(model.parameters())

```

### step 4.Init Booster

```python
plugin = HybridParallelPlugin(tp_size=2, zero_stage=2, pp_size=1, enable_all_optimization=True)
booster = Booster(plugin=plugin)
# You should also pass in your own dataset.
model, dist_optim, criterion, dataloader, _ = booster.boost(model, dist_optim, criterion)
```
### step 5.Train Your Model
```python
steps = 10
for step in range(steps):
    input_ids = torch.ones(1, 100, device="cuda", dtype=torch.int)
    attention_mask = input_ids.clone()
    outputs = model(input_ids.cuda(), attention_mask.cuda())
    loss = criterion(outputs.last_hidden_state)
    booster.backward(loss, dist_optim)
    dist_optim.step()
    dist_optim.zero_grad()
```
### GaLore special handling
For GaLore, we need to specify projection rank for each parameter group and quantization & paged optimizer params. Please refer to bitandbytes for quantization details. Support for ZeRO is underway.
```python
from colossalai.nn.optimizer.galore import get_galore_param_groups
from colossalai.nn.optimizer import DistGaloreAwamW
optim = DistGaloreAwamW(
    get_galore_param_groups(model, decay=1e-2, rank=8),
    lr=lr,
    betas=(beta1, beta2),
    eps=eps,
    nbits=8,
    percentile_clipping=100,
    block_wise=True,
    min_8bit_size=4096,
)
```

## Plugin compatibility
<table>
  <tr>
    <th nowrap="nowrap">Optimizer/Plugin</th>
    <th nowrap="nowrap" align="center">Hybrid Parallel Plugin</th>
    <th nowrap="nowrap" align="center">Low Level Zero Plugin</th>
    <th nowrap="nowrap" align="center">Torch DDP Plugin</th>
    <th nowrap="nowrap" align="center">Gemini Plugin</th>
    <th nowrap="nowrap" align="center">Moe Hybrid Plugin</th>
  </tr>
  <tr>
    <td nowrap="nowrap" align="center" title="Lamb">Lamb</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="center" title="GaLore">GaLore</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="center" title="Adafactor">Adafactor</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="center" title="CAME">CAME</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">✔️</td>
    <td nowrap="nowrap" align="center">❌</td>
    <td nowrap="nowrap" align="center">❌</td>
  </tr>
  <tr>
    <td colspan="39"></td>
  </tr>
</table>

<!-- doc-test-command: colossalai run --nproc_per_node 4 distributed_optimizers.py  -->

## API Reference

{{ autodoc:colossalai.nn.optimizer.distributed_adafactor.DistributedAdaFactor }}
{{ autodoc:colossalai.nn.optimizer.distributed_lamb.DistributedLamb }}
{{ autodoc:colossalai.nn.optimizer.distributed_galore.DistGaloreAwamW }}
{{ autodoc:colossalai.nn.optimizer.distributed_came.DistributedCAME }}

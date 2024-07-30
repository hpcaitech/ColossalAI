# 分布式优化器

Author: Wenxuan Tan, Junwen Duan, Renjie Mao

**相关论文**
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
- [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/pdf/1904.00962)

## 介绍
除了广泛采用的Adam和SGD外，许多现代优化器需要逐层统计信息以有效更新参数，因此无法直接应用于模型层在多个设备上分片的并行设置。我们以提供了优化的分布式实现，，并且通过plugin与Tensor Parallel、DDP和ZeRO无缝集成。
## 优化器
Adafactor 是一种首次采用非负矩阵分解（NMF）的 Adam 变体，用于减少内存占用。CAME 通过引入一个置信度矩阵来改进 NMF 的效果。GaLore 通过将梯度投影到低秩空间，并使用 8 位块状量化进一步减少内存占用。Lamb 允许使用巨大的批量大小而不失准确性，通过按其 Lipschitz 常数的倒数界定的逐层自适应更新实现


## 使用
现在我们展示如何使用分布式 Adafactor 与 booster API 结合 Tensor Parallel 和 ZeRO 2。即使您不使用distributed optimizer，plugin 也会自动将optimizer转换为分布式版本以方便使用。
### step 1. 导包

```python
from transformers import LlamaModel, LlamaConfig
from colossalai.nn.optimizer.distributed_adafactor import DistributedAdaFactor
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
import colossalai
import torch
```

### step 2. 初始化分布式
我们需要先初始化分布式环境. 为了展示, 我们使用 `colossal run --nproc_per_node 4`. 更多初始化方式请参考 [Launch Colossal-AI](../basics/launch_colossalai.md)

```python
colossalai.launch_from_torch()
```

### step 3. 初始化模型和优化器
```python
configuration = LlamaConfig()
model = LlamaModel(configuration).cuda()
criterion = lambda x: x.mean()
dist_optim = DistributedAdaFactor(model.parameters())

```

### step 4.初始化booster和plugin

```python
plugin = HybridParallelPlugin(tp_size=2, zero_stage=2, pp_size=1, enable_all_optimization=True)
booster = Booster(plugin=plugin)
# You should also pass in your own dataset.
model, dist_optim, criterion, dataloader, _ = booster.boost(model, dist_optim, criterion)

```
### step 5.训练
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
### GaLore的特殊初期
对于 GaLore，我们需要为每个参数组指定投影rank，以及量化和分页优化器参数。有关量化的详细信息，请参考 bitandbytes.
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

## 兼容性
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

## API 参考

{{ autodoc:colossalai.nn.optimizer.distributed_adafactor.DistributedAdaFactor }}
{{ autodoc:colossalai.nn.optimizer.distributed_lamb.DistributedLamb }}
{{ autodoc:colossalai.nn.optimizer.distributed_galore.DistGaloreAwamW }}
{{ autodoc:colossalai.nn.optimizer.distributed_came.DistributedCAME }}

# 序列并行

作者: Mingyan Jiang

**前置教程**
- [并行技术](../concepts/paradigms_of_parallelism.md)
- [Booster API](../basics/booster_api.md)
- [Shardformer](../features/shardformer.md)
- [Booster 插件](../basics/booster_plugins.md)

**示例代码**
- [使用序列并行策略](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/llama/benchmark.py)

**相关论文**
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198)
[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889)

## 快速预览

在本教程中，你将学习如何使用序列并行。在 Colossal-AI 中, 我们实现了包括TP+SP， DeepSpeed-Ulysses， ring attention等多种序列并行. 我们下面将介绍如何使用这几种序列并行。

## 目录

在本教程中，我们将介绍三种序列并行的使用:

1. 使用TP+SP；
2. 使用DeepSpeed-Ulysses；
3. 使用ring attention


## Colossal-AI中的实现

在 Colossal-AI 中，shardformer实现了序列并行，并通过`HybridParallelPlugin`和`MoeHybridParallelPlugin`接口可进行调用。相关plugin的介绍请参考plugin的[使用文档](../basics/booster_plugins.md)。

### 使用`HybridParallelPlugin`的序列并行
`HybridParallelPlugin`的序列支持了TP+SP， DeepSpeed-Ulysses， ring attention三种实现，相关序列并行的结束可参考[并行技术介绍文档](../concepts/paradigms_of_parallelism.md)，`HybridParallelPlugin`中的序列并行[例子](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/llama/benchmark.py)

#### 定义模型相关组件

```python
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import torch.distributed as dist
from colossalai.booster import Booster
config = LlamaConfig(max_position_embeddings=4096)
from colossalai.booster.plugin import HybridParallelPlugin

# 定义数据集
class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(
            0, vocab_size, (num_samples, max_length), device=get_accelerator().get_current_device()
        )
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("-s", "--num_steps", type=int, default=5, help="Number of steps to run")
parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
parser.add_argument("--sp", type=int, default=1, help="Sequence parallel size")
args = parser.parse_args()

model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
optimizer = HybridAdam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
# usually, num_samples=args.batch_size * args.num_steps * dp_size
dataset = RandomDataset(
        num_samples=10000, max_length=args.max_length, vocab_size=config.vocab_size
    )
```
### 使用TP+SP
定义plugin,使用该序列并行，`sp_size`会被设置为`tp_size`一致，且tp group 与sp group是重叠的。
```python
plugin = HybridParallelPlugin(
            tp_size=4,
            sp_size=1,
            enable_all_optimization=True,
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="split_gather",
        )
```

启动参数举例：```--tp 2 --sp 8 --sp_mode split_gather```

#### 使用DeepSpeed-Ulysses
定义plugin， 在DeepSpeed-Ulysses的序列并行种，tp group与sp group 是正交的，
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            sp_size=2,
            enable_all_optimization=True,
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="all_to_all",
        )
```
启动参数举例：```--tp 2 --sp 8 --sp_mode all_to_all```

#### 使用ring attention
定义plugin， 在ring attention的序列并行种，tp group与sp group 是正交的，sp_size必须传入准确的并行大小。
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            sp_size=2,
            enable_all_optimization=True,
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="ring_attn",
        )
```
启动参数举例：```--tp 2 --sp 8 --sp_mode ring_attn```

#### 使用booster
```python
booster = Booster(plugin=plugin)
dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, seed=42)
model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)
```

#### 训练模型
```python
for step, batch in enumerate(tqdm(dataloader, desc="Step", disable=not dist.get_rank()==0)):
    outputs = model(**batch)
    loss = outputs[0]
    del outputs  # free memory

    if dist.get_rank() == dist.get_world_size() - 1:
        print(f"Step {step} loss: {loss}")
    booster.backward(loss, optimizer)
    optimizer.step()
    optimizer.zero_grad()
```
### 使用`MoeHybridParallelPlugin`的序列并行
    `MoeHybridParallelPlugin`中的序列并行暂时只支持DeepSpeed-Ulysses类型,使用方法与`HybridParallelPlugin`类似，具体可参考[例子](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/deepseek/benchmark.py)



### 结论
在上述序列并行方法中，ring attention和Ulysses各有优劣，我们需要根据情况来选择合适的序列并行方法：

    通信方面：Ulysses通信量优于ring attention，Ulysess主要包含三次All2All通信量,而ring attention的通信会随着序列长度增长而平方增长。不过另一方面，all2all对底层硬件的要求也会更高。

    内存占用：二者类似。

    模型结构泛化：ring attention优于Ulysses。Ulysses模型泛化性一般，对于head number有要求，需要满足: `head number // (tp group size * sp group size)` ，而ring attention没有此限制。

由于使用简单，对Attention计算不侵入修改，Ulysses目前是序列并行的主流。这些序列并行都可与其他高性能注意力兼容，如flash attention，还可以与ZeRO、TP、PP、DP等多种并行训练策略混合使用。

总的来说，我们更推荐您使用Ulysses，只需要在启动时指定```--sp_mode all_to_all```即可。经过测试，在双机16卡的情况下，使用```--tp 2 --sp 8 --sp_mode all_to_all```的启动参数可以很轻松训练128k长度的序列，同时它的性能表现也是所有序列并行模式中最好的，在双机H800上能够达到约480以上的tflops。但如果追求极致性能优化，或者使用较多机器训练长文本，可以考虑使用ring attention模式的序列并行。


<!-- doc-test-command: torchrun --standalone --nproc_per_node=4 sequence_parallelism.py  -->

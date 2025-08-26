# Sequence Parallelism

Author: Mingyan Jiang

**Prerequisite Tutorials**
- [Paradigms of Parallelism](../concepts/paradigms_of_parallelism.md)
- [Booster API](../basics/booster_api.md)
- [Shardformer](../features/shardformer.md)
- [Booster plugin](../basics/booster_plugins.md)

**Example Code**
- [Using Sequence Parallelism Strategy](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/llama/benchmark.py)

**Related Papers**
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198)
[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889)

## Quick Overview

In this tutorial, you will learn how to use sequence parallelism. In Colossal-AI, we have implemented several types of sequence parallelism, including TP+SP, DeepSpeed-Ulysses, and ring attention. Below, we will introduce how to use these different types of sequence parallelism.

## Table Of Content

In this tutorial, we will cover the use of three sequence parallelism strategies:

1. Using TP+SP;
2. Using DeepSpeed-Ulysses;
3. Using ring attention.


## Implementation in Colossal-AI

In Colossal-AI, sequence parallelism is implemented via the shardformer and can be invoked through the `HybridParallelPlugin` and `MoeHybridParallelPlugin` interfaces. For more information about the plugins, refer to the [plugin usage documentation](../basics/booster_plugins.md).

### Using Sequence Parallelism with HybridParallelPlugin

The `HybridParallelPlugin` supports three types of sequence parallelism: TP+SP, DeepSpeed-Ulysses, and ring attention. You can refer to the parallel techniques introduction [document](../concepts/paradigms_of_parallelism.md) for more details. An [example](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/llama/benchmark.py) of sequence parallelism with HybridParallelPlugin can be found here.

#### Defining Model Components

```python
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import torch.distributed as dist
from colossalai.booster import Booster
config = LlamaConfig(max_position_embeddings=4096)
from colossalai.booster.plugin import HybridParallelPlugin

# define dataset
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
### Using TP+SP
Define the plugin. When using this sequence parallelism, sp_size will be set to match tp_size, and the tp group will overlap with the sp group.
```python
plugin = HybridParallelPlugin(
            tp_size=4,
            sp_size=1,
            enable_all_optimization=True,
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="split_gather",
        )
```
Example of startup command parameters: ```--tp 2 --sp 8 --sp_mode split_gather```

#### Using DeepSpeed-Ulysses
Define the plugin. In the DeepSpeed-Ulysses sequence parallelism, the tp group and sp group are orthogonal.
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            sp_size=2,
            enable_all_optimization=True,
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="all_to_all",
        )
```
Example of startup command parameters: ```--tp 2 --sp 8 --sp_mode all_to_all```

#### Using Ring Attention
Define the plugin. In ring attention sequence parallelism, the tp group and sp group are orthogonal, and sp_size must be set to the correct parallel size.
```python
plugin = HybridParallelPlugin(
            tp_size=2,
            sp_size=2,
            enable_all_optimization=True,
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="ring_attn",
        )
```
Example of startup command parameters: ```--tp 2 --sp 8 --sp_mode ring_attn```

#### Using Booster
```python
booster = Booster(plugin=plugin)
dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, seed=42)
model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)
```

#### Training the Model
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
### Sequence Parallelism with MoeHybridParallelPlugin
Currently, the `MoeHybridParallelPlugin` only supports DeepSpeed-Ulysses sequence parallelism. The usage is similar to HybridParallelPlugin. For specific examples, refer to this [example](https://github.com/hpcaitech/ColossalAI/blob/main/examples/language/deepseek/benchmark.py).



### Conclusion
Among the sequence parallelism methods mentioned, both ring attention and Ulysses have their pros and cons, and we need to choose the appropriate sequence parallelism method based on the situation:

    Communication: Ulysses has lower communication overhead compared to ring attention, as it primarily involves three All-to-All communication ops, whereas the communication cost of ring attention grows quadratically with the sequence length. However, on the other hand, All-to-All op also demands dense network topologies, e.g. NVLink + NVSwitch, so it doesn't scale well across multiple nodes.

    Memory usage: Both are similar in terms of memory consumption.

    Model structure generalization: Ring attention is better than Ulysses in terms of generalization. Ulysses requires that the model config need to meet ```the head number // (tp group size * sp group size)``` condition, while ring attention has no such restrictions.

Due to its simplicity and non-intrusive modification to attention calculation, Ulysses is currently the mainstream for sequence parallelism. All sequence parallel methods can be compatible with other high-performance attention methods such as Flash Attention, and can also be combined with other parallel training strategies like ZeRO, TP, PP, and DP.

Overall, we recommend using Ulysses. You only need to specify ```--sp_mode all_to_all``` during startup. Based on testing, in a two-node, 16-GPU setup, using the startup parameters ```--tp 2 --sp 8 --sp_mode all_to_all```, it's easy to train sequences of up to 128k length, and the performance is the best among all sequence parallelism methods，can reach approximately 480+ TFLOPS on dual H800s. However, if you're aiming for extreme performance optimization or training long texts on a larger scale of machines, you might want to consider using the ring attention.
<!-- doc-test-command: torchrun --standalone --nproc_per_node=4 sequence_parallelism.py  -->

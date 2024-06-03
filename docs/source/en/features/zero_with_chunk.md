# Zero Redundancy Optimizer with chunk-based memory management

Author: [Hongxin Liu](https://github.com/ver217), [Jiarui Fang](https://github.com/feifeibear), [Zijian Ye](https://github.com/ZijianYY)

**Prerequisite:**
- [Train with booster](../basics/booster_api.md)

**Example Code**

- [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt)

**Related Paper**

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://dl.acm.org/doi/10.1145/3394486.3406703)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)

## Introduction

The Zero Redundancy Optimizer (ZeRO) removes the memory redundancies across data-parallel processes by partitioning three
model states (optimizer states, gradients, and parameters) instead of replicating them.
By doing so, memory efficiency is boosted drastically compared to classic data parallelism, while the computational granularity
and communication efficiency is retained.

1. **Shard Optimizer States**: The optimizer states (e.g., for [Adam optimizer](https://arxiv.org/abs/1412.6980), 32-bit weights,
and the first and second momentum estimates) are partitioned across the processes, so that each process updates only its partition.


2. **Shard Gradient**: After reduction inside data parallel process group, gradient tensors are also partitioned such that each process only stores the gradients corresponding to its partition of the optimizer states. Note, Colossal converts gradient into fp32 format to participate in parameter updating.

3. **Shard Parameter**: The 16-bit model parameters are partitioned across the processes of a data parallel group.

4. **[Gemini](../advanced_tutorials/meet_gemini.md)**: Dynamic heterogeneous memory space manager for parameters, gradients and optimizer states.

Besides, this article will introduce the Zero Redundancy Optimizer with chunk-based memory management.

When using ZeRO, we distributed the model by sharding the parameters. The advantage of this method is that the memory of each node is load balanced. But this approach has two significant disadvantages. First, during communication, a temporary memory buffer needs to be allocated and released afterwards, leading to the memory fragmentation problem. Secondly, using tensor as the granularity for communication will cause the network bandwidth underutilized. Generally, the longer the transmitted message length, the higher the bandwidth utilization.

Using the Chunk mechanism introduced in ColossalAI v0.1.8, we can improve the efficiency of ZeRO. We store a continuous set of parameters in initialization order into a Chunk (a chunk is a continuous memory space), and each Chunk has the same size. Organizing memory in chunks can lead to efficient use of network bandwidth between PCI-e and GPU-GPU, reduce the number of communications, and avoid potential memory fragmentation.

Before v0.1.8, ZeRO had a high communication cost for parameter communications. If a parameter was used multiple times in several consecutive operators, there will be repeated communications operations, and the efficiency was highly damaged. This situation is very common when using the Gradient Checkpoint technique, and the parameter will recompute the forward propagation during backward propagation.

Taking GPT as an example, its Checkpoint will be applied to each GPT Block, and each GPT Block contains a Self-Attention layer and an MLP layer. During the backward pass, the forward of the Self-Attention layer and the MLP layer will be computed in turn, and then the backward of the MLP layer and the Self-Attention layer will be computed in turn.

In addition, due to the communication and memory movement of small Tensors, the bandwidth of NVLINK and PCI-E cannot be fully utilized, and each communication and memory movement has the overhead of kernel launch. After using Chunk, multiple small Tensor communication and memory movement can be changed into one large Tensor communication and memory movement, which not only improves bandwidth utilization but also reduces the overhead of kernel launch.

We also provide a lightweight chunk search mechanism to help users automatically find the chunk size with the smallest memory fragmentation.

## Usage

### GeminiDDP

We will use `GeminiDDP` to use ZeRO with chunk-based memory management. This is our new torch.Module wrapper which uses ZeRO-DP and Gemini. ZeRO is for parallelism and Gemini is for memory management.

Gemini allows LazyInitContext, which can save memory when initializing large models with multi-GPUs.

If your model has `N` billion parameters and your GPU memory is `M` GB, we recommend you use LazyInitContext when `4N >= M`. Otherwise, LazyInitContext is optional.

<!--- doc-test-ignore-start -->
```python
with LazyInitContext(default_device=torch.device('cuda')):
  model = gpt2_medium(checkpoint=True)
```
<!--- doc-test-ignore-end -->

We've provided `Booster` API which is user-friendly. We recommend you use `Booster` API. But if you still want to use low level API, you can read below content of this section.

Wrap the model with `GeminiDDP`.

<!--- doc-test-ignore-start -->
```python
model = GeminiDDP(model, hidden_dim=hidden_dim, min_chunk_size_m=min_chunk_size_m)
```
<!--- doc-test-ignore-end -->

`hidden_dim` is the hidden dimension of DNN. Users can provide this argument to speed up searching. If users do not know this argument before training, it is ok. We will use a default value 1024. `min_chunk_size_m` is a floating point, being the minimum chunk size divided by 2^20 (e.g., if min_chunk_size_m=2.5, then the minimum chunk size should be 2.5*(2^20)).If the aggregate size of parameters is still smaller than the minimum chunk size, all parameters will be compacted into one small chunk.

Initialization of the optimizer.
<!--- doc-test-ignore-start -->
```python
optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)
```
<!--- doc-test-ignore-start -->

Training
<!--- doc-test-ignore-start -->
```python
optimizer.zero_grad()
outputs = model(input_ids, attn_mask)
loss = criterion(outputs, input_ids)
optimizer.backward(loss)
optimizer.step()
```
<!--- doc-test-ignore-start -->
> ⚠️ Note: Please do not use `loss.backward()`, the standard way of writing is `optimizer.backward(loss)`.

### Train GPT

In this example, we use `Hugging Face Transformers`. You have to install `transformers` before running this example. We will take `GPT2 Medium` as an example here.

For simplicity, we just use randomly generated data here.

First we only need to import `GPT2LMHeadModel` from `Huggingface transformers` to define our model, which does not require users to define or modify the model, so that users can use it more conveniently.

Define a GPT model:
```python
class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50257,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]

def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)
```

Define our loss function:

```python
class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```


Write a function to get random inputs:

```python
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask
```

Finally, we define a model which uses Gemini + ZeRO DDP and define our training loop, As we pre-train GPT in this example, we just use a simple language model loss:

```python
from colossalai.nn.optimizer import HybridAdam

from colossalai.booster import Booster
from colossalai.lazy import LazyInitContext
from colossalai.booster.plugin import GeminiPlugin

def main():
    args = parse_args()
    BATCH_SIZE = 8
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    NUM_STEPS = 10
    colossalai.launch_from_torch()

    # build criterion
    criterion = GPTLMLoss()
    optimizer = HybridAdam(model.parameters(), lr=0.001)

    torch.manual_seed(123)
    # build GPT model
    with ColoInitContext(default_device=torch.device('cuda')):
      model = gpt2_medium(checkpoint=True)


    # Gemini + ZeRO DP
    plugin = GeminiPlugin(max_norm=1.0, initial_scale=2**5)
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    torch.cuda.synchronize()
    model.train()
    for n in range(NUM_STEPS):
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        booster.backward(loss, optimizer)
        optimizer.step()

    torch.cuda.synchronize()
```
> ⚠️ Note: If you want to use the Gemini module, please do not use the [Gradient Accumulation](../features/gradient_accumulation_with_booster.md) we mentioned before。
The complete example can be found on [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt).

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 zero_with_chunk.py  -->

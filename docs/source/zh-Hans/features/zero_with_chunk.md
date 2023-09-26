# 基于Chunk内存管理的零冗余优化器 (ZeRO)

作者: [Hongxiu Liu](https://github.com/ver217), [Jiarui Fang](https://github.com/feifeibear), [Zijian Ye](https://github.com/ZijianYY)

**前置教程:**

- [booster使用](../basics/booster_api.md)

**示例代码**

- [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt)

**相关论文**

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://dl.acm.org/doi/10.1145/3394486.3406703)
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818)


## 引言

零冗余优化器 (ZeRO) 通过对三个模型状态（优化器状态、梯度和参数）进行划分而不是复制他们，消除了数据并行进程中的内存冗余。该方法与传统的数据并行相比，内存效率得到了极大的提高，而计算粒度和通信效率得到了保留。

1. **分片优化器状态**: 优化器状态 (如 [Adam optimizer](https://arxiv.org/abs/1412.6980), 32位的权重,
以及一二阶动量估计) 被划分到各个进程中, 因此每个进程只更新其分区。


2. **分片梯度**: 在梯度在数据并行进程组内进行 reduction 后, 梯度张量也被划分，这样每个进程只存储与其划分的优化器状态对应的梯度。 注意, Colossal-AI 将梯度转换为 FP32 格式以参与更新参数。

3. **分片参数**: 16位的模型参数被划分到一个数据并行组的进程中。

4. **[Gemini](../advanced_tutorials/meet_gemini.md)**: 对于参数、梯度、优化器状态的动态异构内存空间管理器。

此外，我们还将介绍基于Chunk内存管理的零冗余优化器。

在使用零冗余优化器 (ZeRO)时，我们通过切分参数的方式对模型进行分布式存储，这种方法的优点是每个节点的内存负载是完全均衡的。但是这种方式有很多缺点。首先，通信时需要申请一块临时内存用来通信，通信完毕释放，这回导致存在内存碎片化的问题。其次，以Tensor为粒度进行通信，会导致网络带宽无法充分利用。通常来说传输的消息长度越长带宽利用率越高。

利用ColossalAI v0.1.8引入了Chunk机制，我们可以提升ZeRO的性能。我们将运算顺序上连续的一组参数存入一个Chunk中（Chunk即一段连续的内存空间），每个Chunk的大小相同。Chunk方式组织内存可以保证PCI-e和GPU-GPU之间网络带宽的高效利用，减小了通信次数，同时避免潜在的内存碎片。

在v0.1.8之前，ZeRO在进行参数聚合时通信成本较高，如果一个参数在连续的几次计算中被使用多次，即会发生多次通信，效率较低。这种情况在使用Checkpoint时非常常见，参数在计算backward时会重计算一遍forward。这种情况下，ZeRO的效率便不高。

以GPT为例，其Checkpoint会应用在每一个GPT Block上，每一个GPT Block包含一个Self-Attention层和MLP层。在计算Backward时，会依次计算Self-Attention层、MLP层的forward，然后依次计算MLP层、Self-Attention层的backward。如使用Chunk机制，我们将Self-Attention层和MLP层放在同一个Chunk中，在每个GPT Block的backward的中便无需再通信。

除此之外，由于小Tensor的通信、内存移动没法完全利用NVLINK、PCIE带宽，而且每次通信、内存移动都有kernel launch的开销。使用了Chunk之后可以把多次小Tensor的通信、内存移动变为一次大Tensor的通信、内存移动，既提高了带宽利用，也减小了kernel launch的开销。

我们提供了轻量级的Chunk搜索机制，帮助用户自动找到内存碎片最小的Chunk尺寸。

## 使用

### GeminiDDP

我们将运用`GeminiDDP`的方式来使用基于Chunk内存管理的ZeRO。这是我们新包装的torch.Module ，它使用 ZeRO-DP 和 Gemini，其中ZeRO 用于并行，Gemini 用于内存管理。

Gemini支持惰性初始化, 它可以节省多卡初始化大模型时的显存使用.

如果你的模型有 `N` billion 个参数，你的 GPU 内存为 `M` GB, 当 `4N >= M` 时，我们推荐使用 LazyInitContext。否则，LazyInitContext 是可选的。

<!--- doc-test-ignore-start -->
```python
with LazyInitContext(default_device=torch.device('cuda')):
  model = gpt2_medium(checkpoint=True)
```
<!--- doc-test-ignore-end -->

我们提供了 `Booster` API，它用户友好。我们推荐你使用 `Booster` API。如果您仍然想使用底层 API，您可以继续阅读本节其他内容。

使用 `GeminiDDP` 包装模型。

<!--- doc-test-ignore-start -->
```python
model = GeminiDDP(model, hidden_dim=hidden_dim, min_chunk_size_m=min_chunk_size_m)
```
<!--- doc-test-ignore-end -->

`hidden dim`是DNN的隐藏维度。用户可以提供这个参数来加快搜索速度。如果用户在训练前不知道这个参数也可以。 我们将使用默认值 1024。`min_chunk_size_m`是以兆（2^20）为单位的最小块大小。如果参数的总大小仍然小于最小块大小，则所有参数将被压缩为一个小块。

初始化优化器。
<!--- doc-test-ignore-start -->
```python
optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)
```
<!--- doc-test-ignore-end -->

<!--- doc-test-ignore-start -->
训练
```python
optimizer.zero_grad()
outputs = model(input_ids, attn_mask)
loss = criterion(outputs, input_ids)
optimizer.backward(loss)
optimizer.step()
```
<!--- doc-test-ignore-end -->
> ⚠️ 注意：请不要使用`loss.backward()`，规范写法是`optimizer.backward(loss)`。

### 训练GPT

在此例程中, 我们使用 `Hugging Face Transformers`，并以 `GPT2 Medium` 为例。你必须在允许该例程前安装 `transformers`。

为了简单起见，我们在这里只使用随机生成的数据。

首先我们只需要引入`Huggingface transformers` 的 `GPT2LMHeadModel`来定义我们的模型，不需要用户进行模型的定义与修改，方便用户使用。

定义GPT模型：

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

定义损失函数:

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

写一个获得随机输入的函数:

```python
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask
```


最后，使用booster注入 Gemini + ZeRO DDP 特性, 并定义训练循环。由于我们在这个例子中对GPT进行预训练，因此只使用了一个简单的语言模型损失函数：

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
    colossalai.launch_from_torch(config={})

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
> ⚠️ 注意：如果你使用Gemini模块的话，请不要使用我们之前提到过的[梯度累加](../features/gradient_accumulation_with_booster.md)。
完整的例子代码可以在 [Train GPT with Colossal-AI](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/gpt). 获得。

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 zero_with_chunk.py  -->

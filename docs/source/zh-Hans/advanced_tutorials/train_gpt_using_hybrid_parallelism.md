# 使用混合并行训练 GPT

作者: Hongxin Liu, Yongbin Li

**示例代码**
- [ColossalAI-Examples GPT2](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt_2)
- [ColossalAI-Examples GPT3](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt_3)

**相关论文**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 引言

在上一篇教程中，我们介绍了如何用流水并行训练 ViT。在本教程中，你将学习一个更复杂的场景--用混合并行方式训练GPT。在这种情况下，由于GPT-3过大，即使CPU内存也无法容纳它。因此，你必须自己分割模型。

## 目录

在本教程中，我们将介绍:

1. 基于 colossalai/model_zoo 定义 GPT 模型
2. 处理数据集
3. 使用混合并行训练 GPT

## 导入依赖库

```python
import json
import os
from typing import Callable

import colossalai
import colossalai.utils as utils
import model_zoo.gpt.gpt as col_gpt
import torch
import torch.nn as nn
from colossalai import nn as col_nn
from colossalai.amp import AMP_TYPE
from colossalai.builder.pipeline import partition_uniform
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.schedule import (InterleavedPipelineSchedule,
                                        PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.trainer import Trainer, hooks
from colossalai.utils.timer import MultiTimer
from model_zoo.gpt import GPTLMLoss
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
```



## 定义 GPT 模型

在前面的教程中，我们介绍了3种建立流水并行模型的方法，但对于像 GPT-3 这样的巨大模型，你甚至不能在 CPU 中建立模型。在这种情况下，你必须自己分割模型。

GPT 数据加载器返回 `input_ids` 和 `attention_mask`, 因此我们在 `forward()` 中使用两个关键字参数来获得它们。请注意，对于除第一阶段以外的其他阶段， `forward()` 的第一个位置参数是上一阶段的输出张量。所以 `hidden_states` 来自前一阶段，并且对于第一阶段来说，它是 `None`。

对于 GPT, *word embedding layer* 与 *output head* 共享权重。我们提供 `PipelineSharedModuleWrapper` 在流水阶段间共享参数。它需要一个 `int` 型的 `list` 作为参数, 这意味着 rank 们共享这些参数。你可以使用 `register_module()`
或 `register_parameter()` 来注册一个模块或一个参数作为共享模块或参数。如果你有多组共享模块/参数，你应该有多个 `PipelineSharedModuleWrapper` 实例。 如果参数在**一个**阶段内共享, 你不应该使用
`PipelineSharedModuleWrapper`, 而只是使用同一个模块/参数实例。在这个例子中，*word embedding layer* 在第一阶段, 而 *output head* 在最后一个阶段。因此，他们在 rank `[0, pipeline_size - 1]` 之间共享参数。

对于第一阶段，它维护 embedding layer 和一些 transformer blocks。对于最后一个阶段，它维护一些 transformer blocks 和 output head layer。对于其他阶段，他们只维护一些 transformer blocks。
`partition_uniform(num_layers, pipeline_size, num_chunks)` 返回所有 rank 的 parts, part 是一个 `(start, end)` (不包括end) 的 `tuple`。`start == 0` 表示这是第一阶段, 而 `end == num_layers` 表示这是最后一个阶段。

```python
class PipelineGPTHybrid(nn.Module):
    def __init__(self,
                 num_layers: int = 12,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 vocab_size: int = 50304,
                 embed_drop_rate: float = 0.,
                 act_func: Callable = F.gelu,
                 mlp_ratio: int = 4,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 dtype: torch.dtype = torch.float,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 1e-5,
                 first: bool = False,
                 last: bool = False):
        super().__init__()
        self.embedding = None
        self.norm = None
        self.head = None
        if first:
            self.embedding = col_gpt.GPTEmbedding(
                hidden_size, vocab_size, max_position_embeddings, dropout=embed_drop_rate, dtype=dtype)
        self.blocks = nn.ModuleList([
            col_gpt.GPTBlock(hidden_size, num_attention_heads, mlp_ratio=mlp_ratio, attention_dropout=attn_drop_rate,
                             dropout=drop_rate, dtype=dtype, checkpoint=checkpoint, activation=act_func)
            for _ in range(num_layers)
        ])
        if last:
            self.norm = col_nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.head = col_gpt.GPTLMHead(vocab_size=vocab_size,
                                          dim=hidden_size,
                                          dtype=dtype,
                                          bias=False)

    def forward(self, hidden_states=None, input_ids=None, attention_mask=None):
        if self.embedding is not None:
            hidden_states = self.embedding(input_ids=input_ids)
        batch_size = hidden_states.shape[0]
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)
        if self.norm is not None:
            hidden_states = self.head(self.norm(hidden_states))
        return hidden_states


def build_gpt_pipeline(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    rank = gpc.get_global_rank()
    wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]
    models = []
    for start, end in parts:
        kwargs['num_layers'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = end == num_layers
        logger.info(f'Rank{rank} build layer {start}-{end}, {end-start}/{num_layers} layers')
        chunk = PipelineGPTHybrid(**kwargs).to(device)
        if start == 0:
            wrapper.register_module(chunk.embedding.word_embeddings)
        elif end == num_layers:
            wrapper.register_module(chunk.head)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model


def GPT2_exlarge_pipeline_hybrid(num_chunks=1, checkpoint=False, dtype=torch.float):
    cfg = dict(hidden_size=1600, num_attention_heads=32, checkpoint=checkpoint, dtype=dtype)
    return build_gpt_pipeline(48, num_chunks, **cfg)


def GPT3_pipeline_hybrid(num_chunks=1, checkpoint=False, dtype=torch.float):
    cfg = dict(hidden_size=12288, num_attention_heads=96,
               checkpoint=checkpoint, max_position_embeddings=2048, dtype=dtype)
    return build_gpt_pipeline(96, num_chunks, **cfg)
```

## 处理数据集

我们在这里提供了一个小型 GPT web-text 数据集。 原始格式是 loose JSON, 我们将保存处理后的数据集。

```python
class WebtextDataset(Dataset):
    def __init__(self, path, seq_len=1024) -> None:
        super().__init__()
        root = os.path.dirname(path)
        encoded_data_cache_path = os.path.join(root, f'gpt_webtext_{seq_len}.pt')
        if os.path.isfile(encoded_data_cache_path):
            seq_len_, data, attention_mask = torch.load(
                encoded_data_cache_path)
            if seq_len_ == seq_len:
                self.data = data
                self.attention_mask = attention_mask
                return
        raw_data = []
        with open(path) as f:
            for line in f.readlines():
                raw_data.append(json.loads(line)['text'])
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.unk_token
        encoded_data = tokenizer(
            raw_data, padding=True, truncation=True, max_length=seq_len, return_tensors='pt')
        self.data = encoded_data['input_ids']
        self.attention_mask = encoded_data['attention_mask']
        torch.save((seq_len, self.data, self.attention_mask),
                   encoded_data_cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'input_ids': self.data[index],
            'attention_mask': self.attention_mask[index]
        }, self.data[index]
```

## 使用混合并行训练 GPT

在上一个教程中，我们解释了一些流水并行的参数含义。在本例中，我们可以确定在流水阶段之间交换的每个输出张量的形状。对于 GPT，该形状为
`(MICRO BATCH SIZE, SEQUENCE LEN, HIDDEN SIZE)`。通过设置该参数，我们可以避免交换每个阶段的张量形状。当你不确定张量的形状时，你可以把它保留为
`None`, 形状会被自动推测。请确保你的模型的 `dtype` 是正确的：当你使用 `fp16`，模型的 `dtype` 必须是 `torch.half`；否则，`dtype` 必须是 `torch.float`。对于流水并行，仅支持 `AMP_TYPE.NAIVE`。

你可以通过在 `CONFIG` 里使用 `parallel` 来轻松使用张量并行。数据并行的大小是根据 GPU 的数量自动设置的。

```python
NUM_EPOCHS = 60
SEQ_LEN = 1024
BATCH_SIZE = 192
NUM_CHUNKS = None
TENSOR_SHAPE = (1, 1024, 1600)
# only pipeline parallel
# CONFIG = dict(NUM_MICRO_BATCHES = 192, parallel=dict(pipeline=2), fp16=dict(mode=AMP_TYPE.NAIVE))
# pipeline + 1D model parallel
CONFIG = dict(NUM_MICRO_BATCHES = 192, parallel=dict(pipeline=2, tensor=dict(mode='1d', size=2)), fp16=dict(mode=AMP_TYPE.NAIVE))


def train():
    disable_existing_loggers()
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(config=CONFIG, backend=args.backend)
    logger = get_dist_logger()

    train_ds = WebtextDataset(os.environ['DATA'], seq_len=SEQ_LEN)
    train_dataloader = utils.get_dataloader(train_ds,
                                            seed=42,
                                            batch_size=BATCH_SIZE,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True)

    use_interleaved = NUM_CHUNKS is not None
    num_chunks = 1 if not use_interleaved else NUM_CHUNKS
    model = GPT2_exlarge_pipeline_hybrid(num_chunks=num_chunks, checkpoint=True, dtype=torch.half)
    # model = GPT3_pipeline_hybrid(num_chunks=num_chunks, checkpoint=True, dtype=torch.half)
    if use_interleaved and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    criterion = GPTLMLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-2,)

    engine, train_dataloader, _, _ = colossalai.initialize(model,
                                                           optimizer,
                                                           criterion,
                                                           train_dataloader=train_dataloader)
    global_batch_size = BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    timer = MultiTimer()

    trainer = Trainer(
        engine=engine,
        logger=logger,
        timer=timer
    )

    hook_list = [
        hooks.LossHook(),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False,
    )
```

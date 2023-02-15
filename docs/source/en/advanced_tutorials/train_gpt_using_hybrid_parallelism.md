# Train GPT Using Hybrid Parallelism

Author: Hongxin Liu, Yongbin Li

**Example Code**
- [ColossalAI-Examples GPT2](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt_2)
- [ColossalAI-Examples GPT3](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/gpt_3)

**Related Paper**
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## Introduction

In the previous tutorial, we introduce how to train ViT with pipeline. In this tutorial, you will learn a more complex scenario -- train GPT with hybrid parallelism. In this case, GPT-3 is so large that CPU memory cannot fit it as well. Therefore, you must split the model by yourself.

## Table of content

In this tutorial we will cover:

1. The definition of GPT model, based on colossalai/model_zoo
2. Processing the dataset
3. Training GPT using hybrid parallelism

## Import libraries

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



## Define GPT model

In the previous tutorial, we introduced 3 ways to build a pipelined model. But for huge models like GPT-3, you can't even build the model in CPU. In this case, you must split the model by yourself.

GPT dataloader returns `input_ids` and `attention_mask`, so we use two keyword arguments in `forward()` to get them. Note that for stages except the first stage, the first positional argument of `forward()` is the output tensor from the previous stage. So the `hidden_states` is from the previous stage, and for the first stage it's `None`.

For GPT, the *word embedding layer* shares the weights with the *output head*. We provide `PipelineSharedModuleWrapper` to share parameters among pipeline stages. It takes a `list` of `int` as argument, which means those ranks share the parameters. You can use `register_module()` or `register_parameter()` to register a module or a parameter as the shared module or parameter. If you have multiple sets of shared modules / parameters, you should have multiple `PipelineSharedModuleWrapper` instance. If the parameter is shared within **one** stage, you should not use `PipelineSharedModuleWrapper`, and just use the same module / parameter instance. In this example, the *word embedding layer* is at the first stage, and the *output head* is at the last stage. Thus, they are shared among ranks `[0, pipeline_size - 1]`.

For the first stage, it maintains the embedding layer and some transformer blocks. For the last stage, it maintains some transformer blocks and the output head layer. For other stages, they just maintain some transformer blocks. `partition_uniform(num_layers, pipeline_size, num_chunks)` returns the parts of all ranks, and the part is a `tuple` of `(start, end)` (exclude end). `start == 0` means that it's the first stage, and `end == num_layers` means it's the last stage.

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

## Process the dataset

We provide a small GPT web-text dataset here. The original format is loose JSON, and we will save the processed dataset.

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

## Training GPT using hybrid parallelism

In the previous tutorial, we explained the meanings of some pipeline arguments. In this case, we can determine the shape of each output tensor which is exchanged among pipeline stages. For GPT, the shape is `(MICRO BATCH SIZE, SEQUENCE LEN, HIDDEN SIZE)`. By setting this, we can avoid exchanging the tensor shape of each stage. When you are not sure of the tensor shape, you can just  leave it `None`, and the shape is inferred automatically. Make sure that the `dtype` of your model is correct. When you use `fp16`, the `dtype` of your model must be `torch.half`. Otherwise, the `dtype` must be `torch.float`. For pipeline parallelism, only `AMP_TYPE.NAIVE` is supported.

You can easily use tensor parallel by setting `parallel` in `CONFIG`. The data parallelism size is automatically set based on the number of GPUs.

```python
NUM_EPOCHS = 60
SEQ_LEN = 1024
BATCH_SIZE = 192
NUM_CHUNKS = None
TENSOR_SHAPE = (1, 1024, 1600)
# only pipeline parallel
# CONFIG = dict(parallel=dict(pipeline=2), fp16=dict(mode=AMP_TYPE.NAIVE))
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

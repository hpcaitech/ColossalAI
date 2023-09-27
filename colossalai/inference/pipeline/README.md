# 🐳 Pipeline Inference

## Table of Contents
- [💡 Introduction](#introduction)
- [🔗 Design](#design)
- [🔨 Usage](#usage)
    - [Example](#example)
    - [Quick start](#quick-start)
- [📊 Performance](#performance)

## Introduction

`Pipeline Inference` is a module designed to make inference on a pipeline way. In inference systems, although there is no need to store intermediate information such as activations during forward propagation for backward propagation, the weights of some larger models still cannot fit on a single GPU for inference. This requires us to use model parallelism and other methods to reduce the memory occupation on a single GPU. Pipeline parallelism, as one of the traditional model parallelism approaches, has been widely used due to its reduced all-reduce communication requirements and simple layout. The main issue with pipeline parallelism, known as bubbles, can be almost eliminated in inference because the backward propagation that causes bubbles no longer exists in inference. This makes pipeline parallelism almost bubble-free in the ideal scenario where the sequence length is the same across the pipeline.

## Design

Pipeline Inference is composed of three parts: `PPInferEngine`, `MicroBatchManager` and `generate` [schedule](https://github.com/hpcaitech/ColossalAI/blob/feature/pipeline-infer/colossalai/pipeline/schedule/generate.py).

1. `PPInderEngine` is the High-Level API for users to use. It is responsible for the following tasks:
    - Initialize the pipeline inference environment with `PipelineStageManager` and mdoel with `ShardFormer`.
    - Run the pipeline inference model.

2. `MicroBatchManager` is a structure to manage the micro-batch information. It is responsible for the following tasks:
    - Record each micro-batch information, like generated new tokens and kvcache.
    - Record each micro-batch inference state, like prefill, generate or done.
    - Update the micro-batch information.

3. `generate` schedule implements the simple pipeline inference layout. When pipeline size is 2, we use `torch.distributed.P2Pop` to implement the communication between stages, mainly to solve the race communication. When pipeline size is larger than 2, we use `torch.distributed.broadcast` which is faster than `torch.distributed.P2Pop`.

## Usage

### Example
```python
from colossalai.pipeline import PPInferEngine
# Suppose the pipeline size is 2, and use fp16 to do infenrence. Use Llama as an example.
model = LlamaForCausalLM.from_pretrained('/path/to/model')
inputs = tokenizer("Hello, my dog is cute", "What a good day", return_tensors="pt")
engine = PPInferEngine(
    pp_size=2,
    dtype='fp16',
    micro_batch_size=1,
    new_length=10,
    model=model,
    model_policy=LlamaForCausalLMPipelinePolicy())

output = engine.inference([inputs])

```

### Quick start
```shell
cd benchmark
sh run.sh
```

## Performance

We conducted multiple benchmark tests to evaluate the performance. We compared the inference `latency` and `throughputs` between `Pipeline Inference` and `hugging face` pipeline. The test environment is 2*A10, 20G.

### Llama Throughput(tokens/s)

#### 7b, fp16
| batch_size(micro_batch size)| 2(1) | 4(2) | 8(4) | 16(8) | 32(8) | 32(16)|
| :---: | :---: | :---: | :---: | :---: | :---: | :---:|
| Pipeline Inference(1024, 128) | 33.31 | 59.98 | 98.92 | 143.47 | 152.61 | OOM |
| Hugging Face(1024, 128) |  41.43 | 65.30 | 91.93 | 114.62 | OOM| OOM |
| Pipeline Inference(512, 512) | 43.37 | 82.81 | 148.03 | 229.06 | 238.67 | 312.82 |
| Hugging Face(512, 512) |  49.13 | 84.91 | 132.87 | 178.30 | OOM| OOM |

#### 7b, fp32
| batch_size(micro_batch size)| 2(1) | 4(2) | 8(4) | 16(4) |
| :---: | :---: | :---: | :---: | :---: |
| Pipeline Inference(1024, 128) | 20.61 | 31.23 | 45.20 | 47.46 |
| Hugging Face(1024, 128) | 19.80 | 29.37| OOM | OOM |
| Pipeline Inference(512, 512) | 28.07 | 46.76 | 79.35 | 81.70 |
| Hugging Face(512, 512) |  25.67 | 43.97 | 60.67 | OOM |

#### 13b, fp16
| batch_size(micro_batch size)| 2(1) | 4(2) | 8(4) | 16(4) |
| :---: | :---: | :---: | :---: | :---: |
| Pipeline Inference(1024, 128) | 21.73 | 38.06 | 61.02 | 64.30 |
| Hugging Face(1024, 128) | 23.48 | 37.59 | 53.44 | OOM |
| Pipeline Inference(512, 512) | 26.65 | 49.48 | 86.11 | 88.44 |
| Hugging Face(512, 512) |  27.45 | 47.74 | 74.46 | OOM |

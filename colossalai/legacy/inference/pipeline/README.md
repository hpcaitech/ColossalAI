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
    - Initialize the pipeline inference environment with `PipelineStageManager` and model with `ShardFormer`.
    - Run the pipeline inference model.

2. `MicroBatchManager` is a structure to manage the micro-batch information. It is responsible for the following tasks:
    - Record each micro-batch information, like generated new tokens and kvcache.
    - Record each micro-batch inference state, like prefill, generate or done.
    - Update the micro-batch information.

3. `generate` schedule implements the simple pipeline inference layout. When pipeline size is 2, we use `torch.distributed.P2Pop` to implement the communication between stages, mainly to solve the race communication. When pipeline size is larger than 2, we use `torch.distributed.broadcast` which is faster than `torch.distributed.P2Pop`.

## Usage

### Example
```python
from colossalai.inference import PPInferEngine
from colossalai.inference.pipeline.policies import LlamaModelInferPolicy
import colossalai
from transformers import LlamaForCausalLM, LlamaTokenizer

colossalai.launch_from_torch()

model = LlamaForCausalLM.from_pretrained("/path/to/model")
tokenizer = LlamaTokenizer.from_pretrained("/path/to/model")

# assume the model is inferred with 2 pipeline stages
inferengine = PPInferEngine(pp_size=2, model=model, model_policy=LlamaModelInferPolicy(), new_length=32)

input = ["Introduce a landmark in London","Introduce a landmark in Singapore"]
data = tokenizer(input, return_tensors='pt')
output = inferengine.inference(data.to('cuda'))
print(tokenizer.batch_decode(output))
```

## Performance

We conducted multiple benchmark tests to evaluate the performance. We compared the inference `latency` and `throughputs` between `Pipeline Inference` and `hugging face` pipeline. The test environment is 2 * A10, 20G / 2 * A800, 80G.

### Llama Throughput (tokens/s) | input length=1024, output length=128

#### A10 7b, fp16
| batch_size(micro_batch size) | 2(1)  | 4(2)  |  8(4)  | 16(8)  | 32(8)  | 32(16) |
|:----------------------------:|:-----:|:-----:|:------:|:------:|:------:|:------:|
|      Pipeline Inference      | 40.35 | 77.1  | 139.03 | 232.7  | 257.81 |  OOM   |
|         Hugging Face         | 41.43 | 65.30 | 91.93  | 114.62 |  OOM   |  OOM   |

#### A10 13b, fp16
| batch_size(micro_batch size) | 2(1)  | 4(2)  | 8(4)  | 16(4) |
|:----------------------------:|:-----:|:-----:|:-----:|:-----:|
|      Pipeline Inference      | 25.39 | 47.09 | 83.7  | 89.46 |
|         Hugging Face         | 23.48 | 37.59 | 53.44 |  OOM  |


#### A800 7b, fp16
| batch_size(micro_batch size) | 2(1)  |  4(2)  |  8(4)  | 16(8)  | 32(16) |
|:----------------------------:|:-----:|:------:|:------:|:------:|:------:|
|      Pipeline Inference      | 57.97 | 110.13 | 213.33 | 389.86 | 670.12 |
|         Hugging Face         | 42.44 |  76.5  | 151.97 | 212.88 | 256.13 |


#### A800 13b, fp16
| batch_size(micro_batch size) | 2(1)  | 4(2)  |  8(4)  | 16(8)  | 32(16) |
|:----------------------------:|:-----:|:-----:|:------:|:------:|:------:|
|      Pipeline Inference      | 41.78 | 94.18 | 172.67 | 310.75 | 470.15 |
|         Hugging Face         | 36.57 | 68.4  | 105.81 | 139.51 | 166.34 |

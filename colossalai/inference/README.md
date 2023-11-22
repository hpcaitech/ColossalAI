# Colossal-Inference


## Table of Contents

- 💡 Introduction
- 🔗 Design
- 🗺 Roadmap
- 📊 Performance

## 💡 Introduction

**Colossal-Inference** is the inference module of Colossal-AI, featuring high performance, steady and easy usability. **Colossal-Inference** incorporates the advantages of the latest open-source inference systems, including LightLLM, TGI, FasterTransformer and flash-attention. Additionally, it incorporates design principles from Colossal AI, especially Shardformer, aiming to provide an efficient and scalable solution for large model inference.

## 🔗 Design


### Architecture of inference:

An overview of the Colossal-Inference is below:

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/inference-arch.png" alt="Colossal-Inference" style="zoom:33%;" />

### Components

Colossal-Inference is composed of three main components:

1. High-level inference engine: it allows our inference framework to easily invoke and utilize various parallel methods.
    1. `HybridEngine`: it is a high level interface that integrates with shardformer, especially for multi-card (tensor parallel, pipline parallel) inference
2. Efficient memory management mechanism: Include the key-value cache manager, allowing for zero memory waste during inference.
    1. `cache manager`: serves as a memory manager to help manage the key-value cache, it integrates functions such as memory allocation, indexing and release.
    2. `batch_infer_info`: holds all essential elements of a batch inference, which is updated every batch.
3. High performance kernels and ops: which are inspired from existing libraries and modified correspondingly.

## 🗺 Roadmap

- [x] Design cache manager and batch infer state
- [x] Design TpInference engine to integrates with `Shardformer`
- [x] Register corresponding high-performance `kernel` and `ops`
- [x] Design policies and forwards (e.g. `Llama` and `Bloom`)
    - [x] policy
    - [x] context forward
    - [x] token forward
    - [x] support flash-decoding
- [x] Support all models
    - [x] Llama
    - [x] Llama-2
    - [x] Bloom
    - [x] Chatglm2
- [x] Quantization
    - [x] GPTQ
    - [x] SmoothQuant
- [ ] Benchmarking for all models

## 📊 Performance

### environment:

We conducted multiple benchmark tests to evaluate the performance. We compared the inference `latency` and `throughputs` between `colossal-inference` and original `hugging-face torch fp16`.

For various models, experiments were conducted using multiple batch sizes under the consistent model configuration of `7 billion(7b)` parameters, `1024` input length, and 128 output length. The obtained results are as follows (due to time constraints, the evaluation has currently been performed solely on the `A100` single GPU performance; multi-GPU performance will be addressed in the future):

### Single GPU Performance:

Currently the stats below are calculated based on A100 (single GPU), and we calculate token latency based on average values of context-forward and decoding forward process, which means we combine both of processes to calculate token generation times. We are actively developing new features and methods to further optimize the performance of LLM models. Please stay tuned.

### Tensor Parallelism Inference

##### Llama

|       batch_size        |   8    |   16   |   32   |
| :---------------------: | :----: | :----: | :----: |
| hugging-face torch fp16 | 199.12 | 246.56 | 278.4  |
|   colossal-inference    | 326.4  | 582.72 | 816.64 |

![llama](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Infer-llama7b.png)

#### Bloom

|       batch_size        |   8    |   16   |   32   |
| :---------------------: | :----: | :----: | :----: |
| hugging-face torch fp16 | 189.68 | 226.66 | 249.61 |
|   colossal-inference    | 323.28 | 538.52 | 611.64 |

![bloom](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/Infer-bloom7b.png)


### Pipline Parallelism Inference

We conducted multiple benchmark tests to evaluate the performance. We compared the inference `latency` and `throughputs` between `Pipeline Inference` and `hugging face` pipeline. The test environment is 2 * A10, 20G / 2 * A800, 80G. We set  input length=1024, output length=128.


#### A10 7b, fp16

| batch_size(micro_batch size) | 2(1)  | 4(2)  |  8(4)  | 16(8)  | 32(8)  | 32(16) |
| :--------------------------: | :---: | :---: | :----: | :----: | :----: | :----: |
|      Pipeline Inference      | 40.35 | 77.10 | 139.03 | 232.70 | 257.81 |  OOM   |
|         Hugging Face         | 41.43 | 65.30 | 91.93  | 114.62 |  OOM   |  OOM   |


![ppllama7b](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/pp-a10-llama7b.png)

#### A10 13b, fp16

| batch_size(micro_batch size) | 2(1)  | 4(2)  | 8(4)  | 16(4) |
| :--------------------------: | :---: | :---: | :---: | :---: |
|      Pipeline Inference      | 25.39 | 47.09 | 83.7  | 89.46 |
|         Hugging Face         | 23.48 | 37.59 | 53.44 |  OOM  |

![ppllama13](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/pp-a10-llama13b.png)


#### A800 7b, fp16

| batch_size(micro_batch size) | 2(1)  |  4(2)  |  8(4)  | 16(8)  | 32(16) |
| :--------------------------: | :---: | :----: | :----: | :----: | :----: |
|      Pipeline Inference      | 57.97 | 110.13 | 213.33 | 389.86 | 670.12 |
|         Hugging Face         | 42.44 |  76.5  | 151.97 | 212.88 | 256.13 |

![ppllama7b_a800](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/pp-a800-llama7b.png)

### Quantization LLama

|  batch_size   |   8    |   16   |   32   |
| :-----------: | :----: | :----: | :----: |
|   auto-gptq   | 199.20 | 232.56 | 253.26 |
| smooth-quant  | 142.28 | 222.96 | 300.59 |
| colossal-gptq | 231.98 | 388.87 | 573.03 |

![bloom](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/inference-quant.png)



The results of more models are coming soon!

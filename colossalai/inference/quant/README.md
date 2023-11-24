# Colossal-Inference Quantization


## Table of Contents

- 💡 Introduction
- 🔗 Design
- 🗺 Roadmap
- 📊 Performance

## 💡 Introduction

**Colossal-Inference Quantization** serves as the quantized inference module within the Colossal-AI framework. Quantization is a technique in deep learning aimed at optimizing models by reducing the precision of parameters, thereby decreasing computational demands during inference and improving overall speed and efficiency.  **Colossal-Inference Quantization** encompasses a set of algorithms and tools designed to quantize the weights and activations in deep neural networks. Such quantization modules typically offer user-friendly interfaces, enabling developers to easily apply quantization to their models post-training.

## 🔗 Design


### Architecture of Quantization inference:

An overview of the Quantization is below:

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference/quant-arch.png" alt="Colossal-quant" style="zoom:30%;" />



### Components
- GPTQ

    - The [GPTQ](https://arxiv.org/abs/2210.17323) component requires AutoGPTQ to build quantized weights. The Inference Engine utilizes Shardformer to replace linear, attention, and RMS norm operations.
    - GPTQ Linear includes implementations from exllama and triton. Exllama linear supports only 4-bit weights and is faster than the triton implementation. Triton Linear can support 2 bits and 8 bits.
    - Parallel GPTQ Linear is implemented for tensor parallelism, comprising row linear and column linear implementations.
    - Various optimizations used in the basic inference configuration, such as KV cache, are adopted.

- SmoothQuant

    - [SmoothQuant](https://arxiv.org/abs/2211.10438) implements the process of building quantized models, making it easy to construct quantized models.
    - SmoothQuant utilizes quantized weights and activations. SmoothQuant linear includes three linear functions: W8A8BFP32O32LinearSiLU, W8A8B8O8Linear, W8A8BFP32OFP32Linear, which are used for int8 and fp32 mixed precision.
    - Parallel Linear is implemented for tensor parallelism, including row linear and column linear for the three types of linear functions.
    - To adopt KV cache, SmoothQuant implements the int8 FlashAttention, which is of int8 and fp32 mixed precision. The data type of the KV cache in SmoothQuant is int8, reducing memory consumption.

## 🗺 Roadmap
- [x] Quantization Algorithm
    - [x] GPTQ
    - [x] SmoothQuant
- [x] Support all models
    - [x] Llama
    - [x] Llama-2
    - [ ] Bloom
    - [ ] Chatglm2
- [ ] Benchmarking for all models
    - [ ] Llama
    - [x] Llama-2
    - [ ] Bloom
    - [ ] Chatglm2

## 📊 Performance

### Environment:

We conducted multiple benchmark tests to evaluate the performance. We compared the inference `throughputs` between `colossal-inference` and `AutoGPTQ`.

For various models, experiments were conducted using multiple batch sizes under the consistent model configuration of `7 billion(7b)` parameters. The obtained results are as follows (due to time constraints, the evaluation has currently been performed solely on the `A800` single GPU performance; multi-GPU performance will be addressed in the future):

### Single GPU Performance:

Currently the stats below are calculated based on A800 (single GPU), and we calculate throughputs for ColossalAI GPTQ,  ColossalAI SmoothQuant and AutoGPTQ. We are actively developing new features and methods to further optimize the performance of LLM models. Please stay tuned.

For Llama2-7B, input len = 512, output len = 256

|  batch_size            |   16   |   32   |   64   |
| :--------------------: | :----: | :----: | :----: |
| AutoGPTQ               | 359.60 | 409.50 | 509.93 |
| ColossalAI SmoothQuant | 275.96 | 443.64 | 606.19 |
| ColossalAI GPTQ        | 464.23 | 803.97 | 1256.06 |

For Llama2-7B, input len = 1024, output len = 256

|  batch_size            |   16   |   32   |   64   |
| :--------------------: | :----: | :----: | :----: |
| AutoGPTQ               | 247.64 | 269.60 |   OOM  |
| ColossalAI SmoothQuant | 232.36 | 323.47 | 606.19 |
| ColossalAI GPTQ        | 410.73 | 650.99 | 911.29 |

The results of more models are coming soon!

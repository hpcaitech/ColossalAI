# Inference

We provide an online inference server and a benchmark. We aim to run inference on single GPU, so quantization is essential when using large models.

We support 8-bit quantization (RTN), which is powered by [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [transformers](https://github.com/huggingface/transformers). And 4-bit quantization (GPTQ), which is powered by [gptq](https://github.com/IST-DASLab/gptq) and [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa). We also support FP16 inference.

We only support LLaMA family models now.

## Choosing precision (quantization)

**FP16**: Fastest, best output quality, highest memory usage

**8-bit**: Slow, easier setup (originally supported by transformers), lower output quality (due to RTN), **recommended for first-timers**

**4-bit**: Faster, lowest memory usage, higher output quality (due to GPTQ), but more difficult setup

## Hardware requirements for LLaMA

Tha data is from [LLaMA Int8 4bit ChatBot Guide v2](https://rentry.org/llama-tard-v2).

### 8-bit

|   Model   | Min GPU RAM | Recommended GPU RAM | Min RAM/Swap |           Card examples            |
| :-------: | :---------: | :-----------------: | :----------: | :--------------------------------: |
| LLaMA-7B  |    9.2GB    |        10GB         |     24GB     | 3060 12GB, RTX 3080 10GB, RTX 3090 |
| LLaMA-13B |   16.3GB    |        20GB         |     32GB     |       RTX 3090 Ti, RTX 4090        |
| LLaMA-30B |    36GB     |        40GB         |     64GB     |       A6000 48GB, A100 40GB        |
| LLaMA-65B |    74GB     |        80GB         |    128GB     |             A100 80GB              |

### 4-bit

|   Model   | Min GPU RAM | Recommended GPU RAM | Min RAM/Swap |                       Card examples                        |
| :-------: | :---------: | :-----------------: | :----------: | :--------------------------------------------------------: |
| LLaMA-7B  |    3.5GB    |         6GB         |     16GB     |         RTX 1660, 2060, AMD 5700xt, RTX 3050, 3060         |
| LLaMA-13B |    6.5GB    |        10GB         |     32GB     |     AMD 6900xt, RTX 2060 12GB, 3060 12GB, 3080, A2000      |
| LLaMA-30B |   15.8GB    |        20GB         |     64GB     | RTX 3080 20GB, A4500, A5000, 3090, 4090, 6000, Tesla V100  |
| LLaMA-65B |   31.2GB    |        40GB         |    128GB     | A100 40GB, 2x3090, 2x4090, A40, RTX A6000, 8000, Titan Ada |

## General setup

```shell
pip install -r requirements.txt
```

## 8-bit setup

8-bit quantization is originally supported by the latest [transformers](https://github.com/huggingface/transformers). Please install it from source.

Please ensure you have downloaded HF-format model weights of LLaMA models.

Usage:

```python
import torch
from transformers import LlamaForCausalLM

USE_8BIT = True # use 8-bit quantization; otherwise, use fp16

model = LlamaForCausalLM.from_pretrained(
            "pretrained/path",
            load_in_8bit=USE_8BIT,
            torch_dtype=torch.float16,
            device_map="auto",
        )
if not USE_8BIT:
    model.half()  # use fp16
model.eval()
```

**Troubleshooting**: if you get error indicating your CUDA-related libraries not found when loading 8-bit model, you can check whether your `LD_LIBRARY_PATH` is correct.

E.g. you can set `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`.

## 4-bit setup

Please ensure you have downloaded HF-format model weights of LLaMA models first.

Then you can follow [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa). This lib provides efficient CUDA kernels and weight conversion script.

After installing this lib, we may convert the original HF-format LLaMA model weights to 4-bit version.

```shell
CUDA_VISIBLE_DEVICES=0 python llama.py /path/to/pretrained/llama-7b c4 --wbits 4 --groupsize 128 --save llama7b-4bit.pt
```

Run this command in your cloned `GPTQ-for-LLaMa` directory, then you will get a 4-bit weight file `llama7b-4bit-128g.pt`.

**Troubleshooting**: if you get error about `position_ids`, you can checkout to commit `50287c3b9ae4a3b66f6b5127c643ec39b769b155`(`GPTQ-for-LLaMa` repo).

## Online inference server

In this directory:

```shell
export CUDA_VISIBLE_DEVICES=0
# fp16, will listen on 0.0.0.0:7070 by default
python server.py /path/to/pretrained
# 8-bit, will listen on localhost:8080
python server.py /path/to/pretrained --quant 8bit --http_host localhost --http_port 8080
# 4-bit
python server.py /path/to/pretrained --quant 4bit --gptq_checkpoint /path/to/llama7b-4bit-128g.pt --gptq_group_size 128
```

## Benchmark

In this directory:

```shell
export CUDA_VISIBLE_DEVICES=0
# fp16
python benchmark.py /path/to/pretrained
# 8-bit
python benchmark.py /path/to/pretrained --quant 8bit
# 4-bit
python benchmark.py /path/to/pretrained --quant 4bit --gptq_checkpoint /path/to/llama7b-4bit-128g.pt --gptq_group_size 128
```

This benchmark will record throughput and peak CUDA memory usage.

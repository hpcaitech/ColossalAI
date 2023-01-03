# Train GPT with Colossal-AI

This example shows how to use [Colossal-AI](https://github.com/hpcaitech/ColossalAI) to run huggingface GPT training in distributed manners.

## GPT

We use the [GPT-2](https://huggingface.co/gpt2) model from huggingface transformers. The key learning goal of GPT-2 is to use unsupervised pre-training models to do supervised tasks.GPT-2 has an amazing performance in text generation, and the generated text exceeds people's expectations in terms of contextual coherence and emotional expression.

## Requirements

Before you can launch training, you need to install the following requirements.

### Install PyTorch

```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install [Colossal-AI v0.1.12](https://colossalai.org/download/) From Official Website

```bash
pip install colossalai==0.1.12+torch1.12cu11.3 -f https://release.colossalai.org
```

### Install requirements

```bash
pip install -r requirements.txt
```

This is just an example that we download PyTorch=1.12.0, CUDA=11.6 and colossalai=0.1.12+torch1.12cu11.3. You can download another version of PyTorch and its corresponding ColossalAI version. Just make sure that the version of ColossalAI is at least 0.1.10, PyTorch is at least 1.8.1 and transformers is at least 4.231.
If you want to test ZeRO1 and ZeRO2 in Colossal-AI, you need to ensure Colossal-AI>=0.1.12.

## Dataset

For simplicity, the input data is randonly generated here.

## Training
We provide two solutions. One utilizes the hybrid parallel strategies of Gemini, DDP/ZeRO, and Tensor Parallelism.
The other one uses Pipeline Parallelism Only.
In the future, we are going merge them together and they can be used orthogonally to each other.

### GeminiDPP/ZeRO + Tensor Parallelism
```bash
bash run_gemini.sh
```

The `train_gpt_demo.py` provides three distributed plans, you can choose the plan you want in `run_gemini.sh`. The Colossal-AI leverages Tensor Parallel and Gemini + ZeRO DDP.

- Colossal-AI
- ZeRO1 (Colossal-AI)
- ZeRO2 (Colossal-AI)
- Pytorch DDP
- Pytorch ZeRO


### Pipeline Parallel
```bash
bash run_pp.sh
```


## Performance

Testbed: a cluster of 8xA100 (80GB) and 1xAMD EPYC 7543 32-Core Processor (512 GB). GPUs are connected via PCI-e.
ColossalAI version 0.1.13.

How dose Batch Size affect the efficency.

| model | #GPU | policy | TP | batch per DP | Tflops |
| ---------- | --------- |--------- |--------- |--------- |--------- |
| gpt2_10b |  2  | cpu | 1 | 32 | 122.046 |
| gpt2_10b |  2  | cpu | 1 | 16 | 82.649 |
| gpt2_10b |  2  | cpu | 1 | 8 | 61.354 |


How dose the Placement Policy affect the efficency.

| model | #GPU | policy | TP | batch per DP | Tflops |
| ---------- | --------- |--------- |--------- |--------- |--------- |
| gpt2_10b |  4  | auto | 1 | 8 | 88.657 |
| gpt2_10b |  4  | cuda | 1 | 8 | OOM |
| gpt2_10b |  4  | cpu | 1 | 8 | 61.354 |
| gpt2_10b |  4  | const | 1 | 8 | 82.137 |

How dose the Tensor Parallel Degree affect the efficency.

| model | #GPU | policy | TP | batch per DP | Tflops |
| ---------- | --------- |--------- |--------- |--------- |--------- |
| gpt2_10b |  4  | auto | 1 | 8 | 88.657 |
| gpt2_10b |  4  | auto | 2 | 8 | 56.687 |
| gpt2_10b |  4  | auto | 4 | 8 | 29.019 |
| gpt2_10b |  4  | auto | 4 | 64 | 50.411 |
| gpt2_20b |  1  | cpu | 1 | 8 | 43.102 |
| gpt2_20b |  4  | cpu | 4 | 8 | 28.491 |


Touch the bar of model scale and batch size.

1. `cpu` is the most stable policy for large model and large batch size. One 8 GPU with TP=2, largest batch size of `auto`, `const`
 `cpu` is 64, 32 and 16, respectively.

2. Tensor parallel is necessary for 20B model to reduce model data memory requirement on each GPU.

| model | #GPU | policy | TP | batch per DP | Tflops |
| ---------- | --------- |--------- |--------- |--------- |--------- |
| gpt2_20b |  4  | cpu | 1 | 64 | CUDA OOM |
| gpt2_20b |  4  | auto | 1/2 | 64 | CUDA OOM |
| gpt2_20b |  4  | cpu | 2 | 8 | 43.102 |
| gpt2_20b |  4  | cpu | 2 | 64 | 121.394 |
| gpt2_20b |  8  | auto | 2 | 16 | 99.871 |
| gpt2_20b |  8  | cpu | 2 | 64 | 125.170 |
| gpt2_20b |  8  | const | 2 | 32 | 105.415 |


| model | #GPU | policy | TP | batch per DP | Tflops |
| ---------- | --------- |--------- |--------- |--------- |--------- |
| gpt2_20b |  8  | cpu | 2 | 8 | 46.895 |

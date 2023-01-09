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


## Performance

Testbed: a cluster of 8xA100 (80GB) and 1xAMD EPYC 7543 32-Core Processor (512 GB). GPUs are connected via PCI-e.
ColossalAI version 0.1.13.

[benchmark results on google doc](https://docs.google.com/spreadsheets/d/15A2j3RwyHh-UobAPv_hJgT4W_d7CnlPm5Fp4yEzH5K4/edit#gid=0)

[benchmark results on Tencent doc (for china)](https://docs.qq.com/sheet/DUVpqeVdxS3RKRldk?tab=BB08J2)

### Experimental Features

#### [Pipeline Parallel](./experiments/pipeline_parallel/)
#### [Auto Parallel](./experiments/auto_parallel_with_gpt/)

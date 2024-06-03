# Mixtral

## Usage

### 1. Installation

Please install the latest ColossalAI from source.

```bash
CUDA_EXT=1 pip install -U git+https://github.com/hpcaitech/ColossalAI
```

Then install dependencies.

```bash
cd ColossalAI/applications/ColossalMoE
pip install -e .
```

Additionally, we recommend you to use torch 1.13.1. We've tested our code on torch 1.13.1 and found it's compatible with our code.

### 2. Inference
Yon can use colossalai run to launch inference:
```bash
bash infer.sh
```
If you already have downloaded model weights, you can change name to your weights position in `infer.sh`.

### 3. Train
You first need to create `./hostfile`, listing the ip address of all your devices, such as:
```bash
111.111.111.110
111.111.111.111
```
Then yon can use colossalai run to launch train:
```bash
bash train.sh
```
It requires 16 H100 (80G) to run the training. The number of GPUs should be divided by 8. If you already have downloaded model weights, you can change name to your weights position in `train.sh`.

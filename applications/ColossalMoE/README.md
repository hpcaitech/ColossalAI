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
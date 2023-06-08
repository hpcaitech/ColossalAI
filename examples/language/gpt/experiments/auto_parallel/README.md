# Auto-Parallelism with GPT2

## Requirements

Before you can launch training, you need to install the following requirements.

### Install PyTorch

```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install Colossal-AI

```bash
pip install colossalai==0.2.0
```

### Install transformers

```bash
pip install transformers
```

### Install pulp and coin-or-cbc

```bash
pip install pulp
conda install -c conda-forge coin-or-cbc
```

## Dataset

For simplicity, the input data is randomly generated here.

## Training

```bash
#Run the auto parallel resnet example with 4 GPUs with a dummy dataset.
colossalai run --nproc_per_node 4 auto_parallel_with_gpt.py
```

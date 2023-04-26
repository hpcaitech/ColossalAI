# Auto-Offload Demo with GPT2

## Requirements

Before you can launch training, you need to install the following requirements.

### Install PyTorch

```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install [Colossal-AI v0.2.0](https://colossalai.org/download/) From Official Website

```bash
pip install colossalai==0.2.0+torch1.12cu11.3 -f https://release.colossalai.org
```

### Install transformers

```bash
pip install transformers
```

## Dataset

For simplicity, the input data is randomly generated here.

## Training

```bash
#Run the auto offload on GPT with default setting and a dummy dataset.
bash run.sh
```

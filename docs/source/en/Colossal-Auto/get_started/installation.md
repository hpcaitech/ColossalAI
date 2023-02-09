# Setup

## Announcement

Our auto-parallel feature is a alpha version. It is still under development. We will keep updating it and make it more stable. If you encounter any problem, please feel free to raise an issue.

## Requirements

We need some extra dependencies to support auto-parallel. Please install them before using auto-parallel.

### Install PyTorch

We only support PyTorch 1.12 now, other versions are not tested. We will support more versions in the future.

```bash
#conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
#pip
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install pulp and coin-or-cbc

```bash
pip install pulp
conda install -c conda-forge coin-or-cbc
```

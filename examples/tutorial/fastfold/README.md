# FastFold Inference

## Table of contents

- [Overview](#📚-overview)
- [Quick Start](#🚀-quick-start)
- [Dive into FastFold](#🔍-dive-into-fastfold)

## 📚 Overview

This example lets you to quickly try out the inference of FastFold.

**NOTE: We use random data and random parameters in this example.**


## 🚀 Quick Start

1. Install FastFold

We highly recommend installing an Anaconda or Miniconda environment and install PyTorch with conda.

```
git clone https://github.com/hpcaitech/FastFold
cd FastFold
conda env create --name=fastfold -f environment.yml
conda activate fastfold
python setup.py install
```

2. Run the inference scripts.

```bash
python inference.py --gpus=1 --n_res=256 --chunk_size=None --inplace
```
+ `gpus` means the DAP size
+ `n_res` means the length of residue sequence
+ `chunk_size` introduces a memory-saving technology at the cost of speed, None means not using, 16 may be a good trade off for long sequences.
+ `inplace` introduces another memory-saving technology with zero cost, drop `--inplace` if you do not want it.

## 🔍 Dive into FastFold

There are another features of FastFold, such as:
+ more excellent kernel based on triton
+ much faster data processing based on ray
+ training supported

More detailed information can be seen [here](https://github.com/hpcaitech/FastFold/).

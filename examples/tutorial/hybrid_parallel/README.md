# Multi-dimensional Parallelism with Colossal-AI

## Table of contents

- [Overview](#-overview)
- [Quick Start](#-quick-start)

## 📚 Overview

This example lets you to quickly try out the hybrid parallelism provided by Colossal-AI.
You can change the parameters below to try out different settings in the `config.py`.

```python
# parallel setting
TENSOR_PARALLEL_SIZE = 2
TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    pipeline=2,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)
```

## 🚀 Quick Start

1. Install PyTorch

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Run the training scripts with synthetic data.

```bash
colossalai run --nproc_per_node 4 train.py --config config.py
```

4. Modify the config file to play with different types of tensor parallelism, for example, change tensor parallel size to be 4 and mode to be 2d and run on 8 GPUs.

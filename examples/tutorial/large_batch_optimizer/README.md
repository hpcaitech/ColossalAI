# Large Batch Training Optimization

## Table of contents

- [Large Batch Training Optimization](#large-batch-training-optimization)
  - [Table of contents](#table-of-contents)
  - [📚 Overview](#-overview)
  - [🚀 Quick Start](#-quick-start)

## 📚 Overview

This example lets you to quickly try out the large batch training optimization provided by Colossal-AI. We use synthetic dataset to go through the process, thus, you don't need to prepare any dataset. You can try out the `Lamb` and `Lars` optimizers from Colossal-AI with the following code.

```python
from colossalai.nn.optimizer import Lamb, Lars
```

## 🚀 Quick Start

1. Install PyTorch

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Run the training scripts with synthetic data.

```bash
# run on 4 GPUs
# run with lars
colossalai run --nproc_per_node 4 train.py --config config.py --optimizer lars

# run with lamb
colossalai run --nproc_per_node 4 train.py --config config.py --optimizer lamb
```

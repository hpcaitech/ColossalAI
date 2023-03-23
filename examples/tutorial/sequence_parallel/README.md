# Sequence Parallelism

## Table of contents

- [Sequence Parallelism](#sequence-parallelism)
  - [Table of contents](#table-of-contents)
  - [📚 Overview](#-overview)
  - [🚀 Quick Start](#-quick-start)
  - [🏎 How to Train with Sequence Parallelism](#-how-to-train-with-sequence-parallelism)
    - [Step 1. Configure your parameters](#step-1-configure-your-parameters)
    - [Step 2. Invoke parallel training](#step-2-invoke-parallel-training)

## 📚 Overview

In this tutorial, we implemented BERT with sequence parallelism. Sequence parallelism splits the input tensor and intermediate
activation along the sequence dimension. This method can achieve better memory efficiency and allows us to train with larger batch size and longer sequence length.

Paper: [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)

## 🚀 Quick Start

1. Install PyTorch

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Run with the following command

```bash
export PYTHONPATH=$PWD

# run with synthetic dataset
colossalai run --nproc_per_node 4 train.py
```

> The default config is sequence parallel size = 2, pipeline size = 1, let’s change pipeline size to be 2 and try it again.


## 🏎 How to Train with Sequence Parallelism

We provided `train.py` for you to execute training. Before invoking the script, there are several
steps to perform.

### Step 1. Configure your parameters

In the `config.py` provided, a set of parameters are defined including training scheme, model, etc.
You can also modify the ColossalAI setting. For example, if you wish to parallelize over the
sequence dimension on 8 GPUs. You can change `size=4` to `size=8`. If you wish to use pipeline parallelism, you can set `pipeline=<num_of_pipeline_stages>`.

### Step 2. Invoke parallel training

Lastly, you can start training with sequence parallelism. How you invoke `train.py` depends on your
machine setting.

- If you are using a single machine with multiple GPUs, PyTorch launch utility can easily let you
  start your script. A sample command is like below:

  ```bash
    colossalai run --nproc_per_node <num_gpus_on_this_machine> --master_addr localhost --master_port 29500 train.py
  ```

- If you are using multiple machines with multiple GPUs, we suggest that you refer to `colossalai
  launch_from_slurm` or `colossalai.launch_from_openmpi` as it is easier to use SLURM and OpenMPI
  to start multiple processes over multiple nodes. If you have your own launcher, you can fall back
  to the default `colossalai.launch` function.

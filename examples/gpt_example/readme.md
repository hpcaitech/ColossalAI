# Run GPT With Colossal-AI

## Overview

There are quantities of modes to run GPT in colossal-ai. The `train_gpt.py` script runs training with the specific configuration script in `gpt2_configs/` and `gpt3_configs/` for different parallelisms of GPT-2 and GPT-3, respectively. We have provided some modes of both GPT-2 and GPT-3 and you can modify it to use.

## **USAGE**

```Bash
#!/usr/bin/env sh
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring
export NCCL_P2P_LEVEL=2
export NCCL_NET_GDR_LEVEL=5

export DATA=/path/to/date

torchrun --standalone --nproc_per_node=no_gpus train_gpt.py --config=gpt2_configs/files --from_torch
```

You can copy it and save it as `run.sh` and use `./run.sh` to run the script in your terminal.

Please modify `DATA`, `no_gpus` and `gpt2_configs/files` with the path to your dataset, the number of GPUs and the config file path, respectively.

## GPT-2


Here are the GPT-2 configs' default parameter:

| config       | scale | GPU* | batch  size | MiB of each GPU                                 | TP   | PP   | DP   |
| ------------ | ----- | ---- | ----------- | ----------------------------------------------- | ---- | ---- | ---- |
| gpt2-vanilla | small | 1    | 1           | 6071                                            | 1    | 1    | 1    |
| gpt2-vanilla | small | 2    | 1           | 6637, 6637                                      | 1    | 1    | 2    |
| gpt2-1d      | small | 2    | 1           | 6269, 6269                                      | 2    | 1    | 1    |
| gpt2-2d      | small | 4    | 1           | 6061, 6143, 6167, 6057                          | 4    | 1    | 1    |
| gpt-2.5d     | small | 2    | 1           | 6335, 6347                                      | 2    | 1    | 1    |
| gpt2-3d      | small | 8    | 1           | 6139, 6065, 6167, 6077, 6167, 6077, 6167, 6035  | 8    | 1    | 1    |
| gpt2-pp      | small | 2    | 1           | 5483, 5877                                      | 1    | 2    | 1    |
| gpt2-zero2   | small | 1    | 1           | 4485                                            | 1    | 1    | 1    |
| gpt2-zero3   | small | 1    | 1           | 4701                                            | 1    | 1    | 1    |
| gpt2-nvme    | small | 1    | 1           | 5067                                            | 1    | 1    | 1    |
| gpt2-pp1d    | small | 8    | 8           | 4555, 4571, 5537, 5517, 4555, 4571, 5537, 5517, | 2    | 2    | 2    |

*\*Note: For GPUs, we use Nvidia A100 80G.*

**We set** `TENSOR_PARALLEL` `PIPELINE_PARALLEL` **and** `DATA_PARALLEL` **as small as it can to run every demo with the least number of GPUs.**


### **Modify the config file**

#### **General**

There are some **general rules** when modifying the config files.

```Plain%20Text
TP denotes Tensor Parallel
PP denotes Pipeline Parallel
DP denotes Data Parallel

GPUS = TP * PP * DP
Where DP is autoseted
```

You can set the **batch size** and the **epoch** number by changing the number of `BATCH_SIZE` and 

`NUM_EPOCHS`, respectively.

Then we will introduce the config file of each mode.

Notice `gpt2_zero2.py` and `gpt2_zero3.py` have nothing but `BATCH_SIZE` and `NUM_EPOCHS` to change.

#### **Vanilla & Data Parallel**

`Vanilla` is the basic mode of GPT-2 with no parallel at all. However, if you use more than 1 GPU and TP * PP < no. of GPUs, Colossal-AI will **set DP for you** **automatically**.

#### **1D, 2D, 2.5D, 3D**

In files `gpt2_1d.py, gpt2_2d.py, gpt2_2p5d.py, gpt2_3d.py`, there is a line:

```Python
TENSOR_PARALLEL = 2
```

You can modify it to use more tensor parallel, just with the general rules satisfied.

Specially, `TENSOR_PARALLEL` should be a square number and cubic number for 2D and 3D, respectively.

And `TENSOR_PARALELL / DEPTH` should be a square number for 2.5D, for which the file is 

```
gpt2_2p5d.py
```

#### **Pipeline Parallel**

In `gpt2_pp.py`, there are lines:

```Python
NUM_MICRO_BATCHES = 1
PIPELINE = 2  
# where the type of BATCH_SIZE / NUM_MICRO_BATCHES should be an interger
```

#### **nvme**

If you want to use nvme, run `apt update && apt install libaio-dev` to prepare the environment and change the `nvme_path` in `zero = dict(...)`. Be aware of that `nvme_path` should be the path your local file system.

#### **Pipeline + 1D + Data Parallel**

In `gpt2_pp1d.py`, we have

```Python
BATCH_SIZE = 8
NUM_EPOCHS = 60
NUM_MICRO_BATCHES = 1
HIDDEN_SIZE = 768
PIPELINE = 2
TENSOR_PARALLEL = 2
MODE  = '1d'
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)
```
We have introduced `BATCH_SIZE`, `NUM_EPOCHS`, `NUM_MICRO_BATCHES`, `PIPELINE`, `TENSOR_PARALLEL` above.

`HIDDEN_SIZE` is related to the model, i.e. `gpt2_small` is 768.

You can choose `None, '1d', '2d', '2.5d', '3d'` for `MODE`.

## GPT-3

GPT-3 is a really huge model, for which it seems not possible to train it with a little number of GPUs. Therefore, we choose some common sets of parameters instead of the smallest ones.

Here are our default parameters of GPT-3 configs:

| config         | GPU* | batch size | TP   | PP   | DP   |
| -------------- | ---- | ---------- | ---- | ---- | ---- |
| gpt3_pp1d_min  | 96   | 192        | 4    | 24   | 1    |
| gpt3_pp1d      | 128  | 192        | 4    | 32   | 1    |
| gpt3_pp2d      | 96   | 2*48       | 4    | 24   | 1    |
| gpt3_pp2p5d    | 96   | 2*48       | 4    | 24   | 1    |
| gpt3_zero3_min | 64   | 3          | 1    | 1    | 64   |
| gpt3_zero3     | 96   | 2          | 1    | 1    | 96   |

*\*Note: For GPUs, we use Nvidia A100 40G.*

In the figure above, `_min` means the set of parameters requires the least number of GPUs with the same mode.

GPT-3 and GPT-2 have the same set of parameter.

If you have any question, feel free to raise an issue :)

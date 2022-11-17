# Multi-dimensional Parallelism with Colossal-AI


## 🚀Quick Start
1. Install our model zoo.
```bash
pip install titans
```
2. Run with synthetic data which is of similar shape to CIFAR10 with the `-s` flag.
```bash
colossalai run --nproc_per_node 4 train.py --config config.py -s
```

3. Modify the config file to play with different types of tensor parallelism, for example, change tensor parallel size to be 4 and mode to be 2d and run on 8 GPUs.


## Install Titans Model Zoo

```bash
pip install titans
```


## Prepare Dataset

We use CIFAR10 dataset in this example. You should invoke the `donwload_cifar10.py` in the tutorial root directory or directly run the `auto_parallel_with_resnet.py`.
The dataset will be downloaded to `colossalai/examples/tutorials/data` by default.
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```


## Run on 2*2 device mesh

Current configuration setting on `config.py` is TP=2, PP=2.

```bash
# train with cifar10
colossalai run --nproc_per_node 4 train.py --config config.py

# train with synthetic data
colossalai run --nproc_per_node 4 train.py --config config.py -s
```

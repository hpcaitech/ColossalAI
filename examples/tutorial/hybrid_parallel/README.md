# Multi-dimensional Parallelism with Colossal-AI


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

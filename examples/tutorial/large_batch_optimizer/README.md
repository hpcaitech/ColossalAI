# Comparison of Large Batch Training Optimization

## Prepare Dataset

We use CIFAR10 dataset in this example. You should invoke the `donwload_cifar10.py` in the tutorial root directory or directly run the `auto_parallel_with_resnet.py`.
The dataset will be downloaded to `colossalai/examples/tutorials/data` by default.
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```

You can also use synthetic data for this tutorial if you don't wish to download the `CIFAR10` dataset by adding the `-s` or `--synthetic` flag to the command.


## Run on 2*2 device mesh

```bash
# run with cifar10
colossalai run --nproc_per_node 4 train.py --config config.py

# run with synthetic dataset
colossalai run --nproc_per_node 4 train.py --config config.py -s
```

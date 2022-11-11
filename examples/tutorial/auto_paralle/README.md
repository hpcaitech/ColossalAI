# Handson 3: Auto-Parallelism with ResNet

## Prepare Dataset

We use CIFAR10 dataset in this example. The dataset will be downloaded to `./data` by default. 
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```


## Run on 2*2 device mesh

```bash
colossalai run --nproc_per_node 4 auto_parallel_demo.py
```
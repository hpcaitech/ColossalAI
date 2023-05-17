# Gradient Accumulation

## Prepare Dataset

We use CIFAR10 dataset in this example. The dataset will be downloaded to `./data` by default.
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```

## Verify Gradient Accumulation

To verify gradient accumulation, we can just check the change of parameter values. When gradient accumulation is set, parameters
are only updated in the last step.

```bash
colossalai run --nproc_per_node 1 train.py --config config.py
```

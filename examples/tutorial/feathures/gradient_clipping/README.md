# Gradient Clipping

## Usage

To use gradient clipping, you can just add the following code to your configuration file, and call `clip_grad_by_norm` or `clip_grad_by_value` method of optimizer which after boost if it support clip gradients.

```python
gradient_clipping = <float>
```

## Prepare Dataset

We use CIFAR10 dataset in this example. The dataset will be downloaded to `./data` by default.
If you wish to use customized directory for the dataset. You can set the environment variable `DATA` via the following command.

```bash
export DATA=/path/to/data
```

## Verify Gradient Clipping

To verify gradient clipping, we can just check the change of parameter values.

```bash
colossalai run --nproc_per_node 1 train.py --config config.py
```

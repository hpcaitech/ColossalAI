# Distributed Data Parallel

## 🚀 Quick Start

This example provides a training script and and evaluation script. The training script provides a an example of training ResNet on CIFAR10 dataset from scratch.

- Training Arguments
  - `-r, `--resume`: resume from checkpoint file path
  - `-c`, `--checkpoint`: the folder to save checkpoints
  - `-i`, `--interval`: epoch interval to save checkpoints
  - `-f`, `--fp16`: use fp16

- Eval Arguments
  - `-e`, `--epoch`: select the epoch to evaluate
  - `-c`, `--checkpoint`: the folder where checkpoints are found


### Train

```bash
# train with torch DDP with fp32
colossalai run --nproc_per_node 2 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 2 train.py -c ./ckpt-fp16 --fp16
```

### Eval

```bash
# evaluate fp32 training
python eval.py -c ./ckpt-fp32 -e 80

# evaluate fp16 mixed precision training
python eval.py -c ./ckpt-fp16 -e 80
```

Expected accuracy performance will be:

| Model     | Single-GPU Baseline FP32 | Booster DDP with FP32 | Booster DDP with FP16 |
| --------- | ------------------------ | --------------------- | --------------------- |
| ResNet-18 | 85.85%                   | 85.03%                | 85.12%                |

**Note: the baseline is a adapted from the [script](https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/) to use `torchvision.models.resnet18`**

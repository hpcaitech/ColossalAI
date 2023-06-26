#!/bin/bash
set -xe

export DATA=/data/scratch/cifar-10

pip install -r requirements.txt

for plugin in "torch_ddp" "torch_ddp_fp16" "low_level_zero"; do
    colossalai run --nproc_per_node 4 train.py --interval 0 --target_acc 0.84 --plugin $plugin
done

#!/usr/bin/env sh

## phase 1: self-supervised training (both single- and multi- GPU training strategies are provided)
# single-GPU training
CUDA_VISIBLE_DEVICES=0 python train_simclr.py --local_rank=0 --world_size=1 --port=29562 --config=cifar_simclr.py
# multi-GPU training
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_simclr.py --world_size 4 --config cifar_simclr.py

## phase 2: linear evaluation
CUDA_VISIBLE_DEVICES=0 python train_linear.py --local_rank=0 --world_size=1 --port=29520 --config=le_cifar_simclr.py
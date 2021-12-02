#!/usr/bin/env sh

# single-GPU training
CUDA_VISIBLE_DEVICES=2 python train_simclr.py --local_rank=0 --world_size=1 --port=29592 --config=cifar_simclr.py

# multi-GPU training
# CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node 4 train_simclr.py --world_size 4 --config cifar_simclr.py

# linear evaluation
# CUDA_VISIBLE_DEVICES=1 python train_linear.py --local_rank=0 --world_size=1 --port=29514 --config=le_cifar_simclr.py
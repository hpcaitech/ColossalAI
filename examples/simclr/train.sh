#!/usr/bin/env sh

# train_simclr.py: main script for SimCLR self-supervised training
# train_linear.py: main script for linear evaluation
# cifar_simclr.py: config file for SimCLR self-supervised training
# le_cifar_simclr.py: config file for linear evaluation


# single-GPU training
CUDA_VISIBLE_DEVICES=0 python train_simclr.py --local_rank=0 --world_size=1 --port=29586 --config=cifar_simclr.py

# multi-GPU training
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train_simclr.py --world_size 4 --config cifar_simclr.py

# linear evaluation
# CUDA_VISIBLE_DEVICES=1 python train_linear.py --local_rank=0 --world_size=1 --port=29512 --config=le_cifar_simclr.py
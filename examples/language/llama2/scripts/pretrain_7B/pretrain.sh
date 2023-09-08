#!/bin/bash

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts.txt)

cd ../..

export CUDA_VISIBLE_DEVICES=4,5,6,7
export TORCH_CUDA_ALLOC_HOST_RESERVED=0
torchrun --standalone --nproc_per_node 4 train.py \
    --mode "pretrain" \
    --plugin "hybrid_parallel" \
    --config "7b" \

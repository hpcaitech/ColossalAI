#!/bin/bash

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts.txt)

cd ../..

torchrun --standalone --nproc_per_node 4 train.py \
    --mode "pretrain" \
    --plugin "hybrid_parallel" \
    --config "7b" \

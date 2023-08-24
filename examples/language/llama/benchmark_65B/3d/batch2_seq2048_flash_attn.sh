#!/bin/bash

################
#Load your environments and modules here
################


cd ../..

# NCCL IB environment variables
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# 4-tp + 4-pp + 2-zero1, num microbatches=8
colossalai run --nproc_per_node 8 --hostfile YOUR_HOST_FILE --master_addr YOUR_MASTER_ADDR benchmark.py -c '65b' --plugin "3d" -l 2048 -g -b 16 -x --tp 4 --pp 4 --zero 1 --mbs 8

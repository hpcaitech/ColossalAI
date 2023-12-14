#!/bin/bash

set -xue

export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export GLOO_SOCKET_IFNAME=eth0

MODEL="8b"
BATCH_SIZE=1
SEQ_LENGTH=2048
WARMUP=8
ACTIVE=4

# HACK: make model importable
example_dir=$(dirname $(realpath $(dirname $0)))
if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=$example_dir
else
    export PYTHONPATH=$example_dir:$PYTHONPATH
fi

# single node
torchrun --standalone $example_dir/benchmark/benchmark_fsdp.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE

# multi node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=node_rank --master_addr=master_addr --master_port=master_port \
    $example_dir/benchmark/benchmark_fsdp.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE

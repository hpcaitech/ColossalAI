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

NUM_GPU=8
MODEL="8b"
SEQ_LENGTH=2048
WARMUP=20
ACTIVE=4

# HACK: make model importable
example_dir=$(dirname $(realpath $(dirname $0)))
if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=$example_dir
else
    export PYTHONPATH=$example_dir:$PYTHONPATH
fi


# ep
echo -e "\n\n Naive EP \n\n"
colossalai run --nproc_per_node $NUM_GPU --hostfile "hostfile.txt" \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 12 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep \
    --zero_stage 2


# ep_zero
echo -e "\n\n EP-ZERO \n\n"
colossalai run --nproc_per_node $NUM_GPU --hostfile "hostfile.txt" \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 20 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep_zero \
    --use_kernel \
    --extra_dp_size 2 \
    --zero_stage 1 \
    --load_balance \
    --overlap_alltoall

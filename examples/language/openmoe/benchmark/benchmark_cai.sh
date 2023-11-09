#!/bin/bash

set -xue

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
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 8 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep \
    --zero_stage 2


# ep_zero
echo -e "\n\n EP-ZERO \n\n"
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 16 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep_zero \
    --use_kernel \
    --extra_dp_size 2 \
    --zero_stage 1 \
    --load_balance

echo -e "\n\n EP-ZERO + Overlap \n\n"
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 16 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep_zero \
    --use_kernel \
    --extra_dp_size 2 \
    --zero_stage 1 \
    --load_balance \
    --overlap_alltoall


# hybrid
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 128 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --use_kernel \
    --plugin hybrid \
    --pp_size 2 \
    --dp_size 1 \
    --ep_size 4 \
    --zero_stage 1 \
    --microbatch_size 32

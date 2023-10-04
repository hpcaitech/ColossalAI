#!/bin/bash

set -xue

NUM_GPU=8
MODEL="8b"
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

# zero2
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 4 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero2 \
    --use_kernel

# zero2_tp
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 12 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero2_tp \
    --use_kernel

# zero2_ep
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 12 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero2_ep \
    --use_kernel

# hybrid
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 512 \
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

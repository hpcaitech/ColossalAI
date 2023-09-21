#!/bin/bash

set -xue

NUM_GPU=4
MODEL="base"
BATCH_SIZE=1
SEQ_LENGTH=2048
WARMUP=10
ACTIVE=10

# HACK: make model importable
example_dir=$(dirname $(realpath $(dirname $0)))
if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=$example_dir
else
    export PYTHONPATH=$example_dir:$PYTHONPATH
fi

# hybrid
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --use_kernel \
    --plugin hybrid \
    --pp_size 2 \
    --dp_size 1 \
    --ep_size 2 \
    --zero_stage 1 \
    --microbatch_size 1

# zero1
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero1 \
    --use_kernel

# zero2
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero2 \
    --use_kernel

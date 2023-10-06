#!/bin/bash

set -xue

NUM_GPU=4
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

# zero
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 4 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero \
    --use_kernel

# ep
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 12 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep \
    --use_kernel \
    --extra_dp_size 2

# ep_zero
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 12 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep_zero \
    --use_kernel \
    --extra_dp_size 2

# zero_ep
torchrun --standalone --nproc_per_node $NUM_GPU \
    $example_dir/benchmark/benchmark_cai.py \
    --model_name $MODEL \
    --batch_size 12 \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin zero_ep \
    --use_kernel \
    --extra_dp_size 2

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

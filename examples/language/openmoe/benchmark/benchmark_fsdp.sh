#!/bin/bash

set -xue

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

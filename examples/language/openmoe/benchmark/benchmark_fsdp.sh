#!/bin/bash

set -xue

NUM_GPU=8
MODEL="8b"
BATCH_SIZE=1
SEQ_LENGTH=2048
WARMUP=6
ACTIVE=3

# HACK: make model importable
example_dir=$(dirname $(realpath $(dirname $0)))
if [ -z ${PYTHONPATH+x} ]; then
    export PYTHONPATH=$example_dir
else
    export PYTHONPATH=$example_dir:$PYTHONPATH
fi

python $example_dir/benchmark/benchmark_fsdp.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE

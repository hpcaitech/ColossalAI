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

python $example_dir/benchmark/benchmark_fsdp.py \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE

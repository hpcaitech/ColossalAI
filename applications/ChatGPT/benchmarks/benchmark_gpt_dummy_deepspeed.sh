#!/usr/bin/env bash
# Usage: $0 <?number-of-gpus> <?strategy> <?model>
set -xu

BASE=$(realpath $(dirname $0))


PY_SCRIPT=${BASE}/benchmark_gpt_dummy_deepspeed.py
export OMP_NUM_THREADS=8

function tune_batch_size() {
    # we found when experience batch size is equal to train batch size
    # peak CUDA memory usage of making experience phase is less than or equal to that of training phase
    # thus, experience batch size can be larger than or equal to train batch size
    for bs in 1 2 4 8 16 32 64 128 256; do
        deepspeed --num_gpus $1 $PY_SCRIPT --model $2 --strategy $3 --experience_batch_size $bs --train_batch_size $bs || return 1
    done
}

if [ $# -eq 0 ]; then
    num_gpus=(1 2 4 8)
else
    num_gpus=($1)
fi

if [ $# -le 1 ]; then
    strategies=('deepspeed' 'deepspeed_zero1' 'deepspeed_zero2' 'deepspeed_zero3' 'deepspeed_zero1_cpu' 'deepspeed_zero2_cpu' 'deepspeed_zero3_cpu')
else
    strategies=($2)
fi

if [ $# -le 2 ]; then
    models=("s" "m" "l" "xl" "2b" "4b" "6b" "8b" "10b")
else
    models=($3)
fi


for num_gpu in ${num_gpus[@]}; do
    for strategy in ${strategies[@]}; do
        for model in ${models[@]}; do
            tune_batch_size $num_gpu $model $strategy || break
        done
    done
done

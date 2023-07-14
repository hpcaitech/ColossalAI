#!/bin/bash

set -xue

echo "Hint: You can run this script with 'verbose' as the first argument to run all strategies."

if [[ $# -ne 0 && "$1" == "verbose" ]]; then
    STRATEGIES=(
        'ddp'
        'colossalai_gemini'
        'colossalai_gemini_cpu'
        'colossalai_zero2'
        'colossalai_zero2_cpu'
        'colossalai_zero1'
        'colossalai_zero1_cpu'
    )
else
    STRATEGIES=(
        'colossalai_zero2'
    )
fi

BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
BENCHMARKS_DIR=$BASE_DIR/benchmarks

echo "[Test]: testing benchmarks ..."

for strategy in ${STRATEGIES[@]}; do
    torchrun --standalone --nproc_per_node 1 $BENCHMARKS_DIR/benchmark_opt_lora_dummy.py \
        --model 125m --critic_model 125m --strategy ${strategy} --lora_rank 4 \
        --num_episodes 2 --num_collect_steps 4 --num_update_steps 2 \
        --train_batch_size 2 --experience_batch_size 4
done

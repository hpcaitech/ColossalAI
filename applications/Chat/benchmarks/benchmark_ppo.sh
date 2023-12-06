#!/usr/bin/env bash

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 8

set -xu

NUM_RETRY=3
BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples
TEMP_DIR=$BASE_DIR/temp
MODEL_SAVE_PATH=$TEMP_DIR/rlhf_models
MODELS_DIR=$TEMP_DIR/models_config
MODELS=('125m' '350m' '700m' '1.3b' '2.7b' '3.5b' '5.5b' '6.7b' '10b' '13b')
PLUGINS=('zero2', 'zero2_cpu', '3d')
LORA_RANK=('0', '20')

export OMP_NUM_THREADS=8

rm ./benchmark_memory_consumption.txt
rm ./benchmark_performance_summarization.txt

# install requirements
pip install -r $EXAMPLES_DIR/requirements.txt

random_choice() {
    local arr=("$@")
    local len=${#arr[@]}
    local idx=$((RANDOM % len))
    echo ${arr[$idx]}
}

echo "[Test]: testing ppo ..."

SKIPPED_TESTS=(
)

GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        plugins=($(shuf -e "${PLUGINS[@]}"))
        for plugin in ${plugins[@]}; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$model
            tokenizer_dir="facebook/opt-125m"
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            if [[ $plugin == "3d" ]]; then
                tp='4'
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a prompt_dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    prompt_dataset+=("$TEMP_DIR/rlhf_data/tokenized_opt_prompt/arrow/part-$split")
                done
                declare -a ptx_dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    ptx_dataset+=("$TEMP_DIR/rlhf_data/tokenized_opt_ptx/arrow/part-$split")
                done
                colossalai run --nproc_per_node 8 --master_port 28547 $BASE_DIR/benchmarks/benchmark.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --prompt_dataset ${prompt_dataset[@]} \
                    --pretrain_dataset ${ptx_dataset[@]} \
                    --ptx_batch_size 1 \
                    --ptx_coef 0.2 \
                    --save_path $MODEL_SAVE_PATH \
                    --lora_rank $lora_rank \
                    --plugin $plugin \
                    --num_episodes 5 \
                    --num_collect_steps 1 \
                    --num_update_steps 4 \
                    --max_seq_len 1024 \
                    --max_length 2048 \
                    --experience_batch_size 4 \
                    --train_batch_size 1 \
                    --accumulation_steps 32 \
                    --lr 9e-6 \
                    --mixed_precision "bf16" \
                    --grad_clip 1.0 \
                    --tp $tp \
                    --lr 2e-5 \
                    --use_flash_attn \
                    $grad_ckpt
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf $MODEL_SAVE_PATH/*
                    rm -rf $MODELS_DIR/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done

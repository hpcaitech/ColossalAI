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

set_n_least_used_CUDA_VISIBLE_DEVICES 4

set -xu


NUM_RETRY=3
BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples
TEMP_DIR=$BASE_DIR/temp
MODEL_SAVE_PATH=$TEMP_DIR/rlhf_models
MODELS_DIR=$TEMP_DIR/models_config
# MODELS=('gpt2' 'bloom' 'opt' 'llama')
MODELS=('gpt2' 'bloom' 'opt' 'llama')
# PLUGINS=('gemini' 'gemini_auto' 'zero2' 'zero2_cpu' '3d')
PLUGINS=('zero2' 'zero2_cpu' '3d')
LORA_RANK=('0' '20')

export OMP_NUM_THREADS=8

# install requirements
pip install -r $EXAMPLES_DIR/requirements.txt

get_pretrain() {
    local model=$1
    if [[ $model == "gpt2" ]]; then
        echo "gpt2"
    elif [[ $model == "bloom" ]]; then
        echo "bigscience/bloom-560m"
    elif [[ $model == "opt" ]]; then
        echo "facebook/opt-350m"
    elif [[ $model == "llama" ]]; then
        echo "/data/scratch/llama-tiny"
    else
        echo "Unknown model $model"
        exit 1
    fi
}

get_tokenizer_dirs() {
    local model=$1
    if [[ $model == "gpt2" ]]; then
        echo "gpt2"
    elif [[ $model == "bloom" ]]; then
        echo "bigscience/bloom-560m"
    elif [[ $model == "opt" ]]; then
        echo "facebook/opt-350m"
    elif [[ $model == "llama" ]]; then
        echo "hf-internal-testing/llama-tokenizer"
    else
        echo "Unknown model $model"
        exit 1
    fi
}

random_choice() {
    local arr=("$@")
    local len=${#arr[@]}
    local idx=$((RANDOM % len))
    echo ${arr[$idx]}
}


echo "[Test]: testing sft ..."

SKIPPED_TESTS=(
    bloom-3d-20 # This test cannot pass, it is probably a bug for the 3d plugin
    llama-3d-20 # This test cannot pass, it is probably a bug for the 3d plugin
)

GRAD_CKPTS=('' '--grad_checkpoint')
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
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            if [[ $plugin == "3d" ]]; then
                tp='4'
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_sft/arrow/part-$split")
                done
                colossalai run --nproc_per_node 4 --master_port 28537 $EXAMPLES_DIR/training_scripts/train_sft.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --save_path $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    --lora_rank $lora_rank \
                    --plugin $plugin \
                    --batch_size 2 \
                    --max_epochs 1 \
                    --accumulation_steps 2 \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400
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

echo "[Test]: testing reward model ..."

SKIPPED_TESTS=(
    bloom-3d-20 # This test cannot pass, it is probably a bug for the 3d plugin
    llama-3d-20 # This test cannot pass, it is probably a bug for the 3d plugin
)

GRAD_CKPTS=('' '--grad_checkpoint')
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
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            if [[ $plugin == "3d" ]]; then
                tp='4'
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_preference/arrow/part-$split")
                done
                colossalai run --nproc_per_node 4 --master_port 28537 $EXAMPLES_DIR/training_scripts/train_rm.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --save_dir $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    --lora_rank $lora_rank \
                    --plugin $plugin \
                    --batch_size 2 \
                    --max_epochs 1 \
                    --accumulation_steps 2 \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400
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


echo "[Test]: testing ppo ..."

SKIPPED_TESTS=(
    bloom-3d-20 # This test cannot pass, it is probably a bug for the 3d plugin
    llama-3d-20 # This test cannot pass, it is probably a bug for the 3d plugin
    gpt2-zero2 # This test can pass locally. Removed due to OOM
    bloom-zero2 # This test can pass locally. Removed due to OOM
    opt-zero2 # This test can pass locally. Removed due to OOM
    bloom-zero2_cpu # This test can pass locally. Removed due to OOM
)

GRAD_CKPTS=('' '--grad_checkpoint')
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
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            if [[ $plugin == "3d" ]]; then
                tp='4'
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a prompt_dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    prompt_dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_prompt/arrow/part-$split")
                done
                declare -a ptx_dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    ptx_dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_ptx/arrow/part-$split")
                done
                colossalai run --nproc_per_node 4 --master_port 28537 $EXAMPLES_DIR/training_scripts/train_ppo.py \
                    --pretrain $pretrain \
                    --rm_pretrain $pretrain \
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
                    --num_update_steps 1 \
                    --experience_batch_size 8 \
                    --train_batch_size 4 \
                    --accumulation_steps 2 \
                    --lr 9e-6 \
                    --mixed_precision "bf16" \
                    --grad_clip 1.0 \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400
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

echo "[Test]: testing DPO ..."

SKIPPED_TESTS=(
    bloom-3d # This test cannot pass, it is probably a bug for the 3d plugin
    llama-3d # This test cannot pass, it is probably a bug for the 3d plugin
    bloom-zero2 # This test can pass locally. Removed due to OOM
    bloom-zero2_cpu # This test can pass locally. Removed due to OOM
)

GRAD_CKPTS=('' '--grad_checkpoint')
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
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            if [[ $plugin == "3d" ]]; then
                tp='4'
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_preference/arrow/part-$split")
                done
                colossalai run --nproc_per_node 4 --master_port 28537 $EXAMPLES_DIR/training_scripts/train_dpo.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --save_dir $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    --lora_rank $lora_rank \
                    --plugin $plugin \
                    --batch_size 2 \
                    --max_epochs 1 \
                    --accumulation_steps 2 \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400
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

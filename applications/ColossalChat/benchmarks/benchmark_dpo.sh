#!/bin/bash
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

PROJECT_NAME="dpo"
PARENT_CONFIG_FILE="./benchmark_config" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="" # huggingface or local tokenizer path

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 4 --master_port 31313 benchmark_dpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --config_file $CONFIG_FILE \
    --plugin "zero2_cpu" \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --batch_size 8 \
    --lr 1e-6 \
    --beta 0.1 \
    --gamma 0.6 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --max_length 2048 \
    --dataset_size 640 \
    --weight_decay 0.01 \
    --warmup_steps 60 \
    --disable_reference_model \
    --length_normalization \
    --grad_checkpoint \
    --use_flash_attn

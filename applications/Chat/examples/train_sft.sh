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
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODEL_DIR="/home/zhongyuting/model/Colossal-Llama-2-7b-base"
DATASET_DIR="/home/jiangmingyan/workspace/chat-data"
OUTPUT_DIR="./output"

PRETRAIN_MODEL_DIR=${MODEL_DIR}
TRAINING_DATASET_DIR=${DATASET_DIR}/data.json
TENSORBOARD_OUTPUT_DIR=${OUTPUT_DIR}/tensorboard
CHECKPOINT_SAVE_DIR=${OUTPUT_DIR}/checkpoint
SAVE_PRETRAINED_MODEL_PATH=${CHECKPOINT_SAVE_DIR}

mkdir -p ${TENSORBOARD_OUTPUT_DIR}
mkdir -p ${SAVE_PRETRAINED_MODEL_PATH}
echo ${PRETRAIN_MODEL_DIR}

export CUDA_LAUNCH_BLOCKING=1

torchrun --nnodes 1 --nproc_per_node 4 --master_port 31312 train_sft.py \
    --strategy "zero2" \
    --model "llama2" \
    --tokenizer ${PRETRAIN_MODEL_DIR} \
    --pretrain ${PRETRAIN_MODEL_DIR} \
    --dataset ${TRAINING_DATASET_DIR} \
    --max_datasets_size 125 \
    --save_path ${SAVE_PRETRAINED_MODEL_PATH} \
    --max_epochs 1 \
    --batch_size 1 \
    --max_len 512 \
    --lora_rank 0 \
    --log_interval 10 \
    --lr 2e-5 \
    --accumulation_steps 8 \
    --log_dir ${TENSORBOARD_OUTPUT_DIR} \
    --use_wandb \
    --grad_checkpoint
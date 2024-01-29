#!/bin/bash

PROJECT_NAME="training_7b_npu"
PARENT_SAVE_DIR="path_to_checkpoint_folder"
PARENT_TENSORBOARD_DIR="path_to_tensorboard_folder"
PARENT_CONFIG_FILE="path_to_config_folder"
PRETRAINED_MODEL_PATH="path_to_Colossal-LLaMA-2-7b-base_weights"

declare -a dataset=(
    # path to pretokenized dataset folder with .arrow data files 
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

torchrun --nproc_per_node 8 --master_port 30013 train_sft_npu.py \
    --pretrained $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 300 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 1 \
    --accumulation_steps 32 \
    --micro_batch_size 2 \
    --lr 1e-4 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0 \
    --use_grad_checkpoint \
    --xformers \
    --use_neft
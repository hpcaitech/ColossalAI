#!/bin/bash

PROJECT_NAME="training_npu_7b_new"
PARENT_SAVE_DIR="/home/lczyt/training/training_npu_7b_new/checkpoint/"
PARENT_TENSORBOARD_DIR="/home/lczyt/training/training_npu_7b_new/tensorboard/"
PARENT_CONFIG_FILE="/home/lczyt/training/training_npu_7b_new/config/"
PRETRAINED_MODEL_PATH="/home/lczyt/models/Colossal-LLaMA-2-7b-base"

declare -a dataset=(
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00000
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00001
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00002
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00003
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00004
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00005
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00006
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00007
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00008
    /home/lczyt/sft_data/7b_tokenized/arrow/part-00009
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
    --num_epochs 3 \
    --accumulation_steps 32 \
    --micro_batch_size 2 \
    --lr 1e-4 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0 \
    --use_grad_checkpoint \
    --xformers \
    --use_neft
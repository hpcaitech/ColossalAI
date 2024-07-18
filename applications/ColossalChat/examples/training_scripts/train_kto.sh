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

PROJECT_NAME="kto"
PARENT_SAVE_DIR="/home/nvme-share/home/yeanbang/data/experiments/kto/checkpoint" # Path to a folder to save checkpoints
PARENT_TENSORBOARD_DIR="/home/nvme-share/home/yeanbang/data/experiments/kto/log" # Path to a folder to save logs
PARENT_CONFIG_FILE="/home/nvme-share/home/yeanbang/data/experiments/kto/log" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/home/nvme-share/home/yeanbang/data/model/hh_rlhf_sheared_llamasft-2024-07-17-07-29-29/modeling" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="/home/nvme-share/share/models/Sheared-LLaMA-1.3B" # huggingface or local tokenizer path

declare -a dataset=(
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00000
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00001
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00002
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00003
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00004
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00005
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00006
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00007
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00008
    /home/nvme-share/home/yeanbang/data/experiments/kto/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 4 --master_port 31313 train_kto.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 1000 \
    --save_dir $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --batch_size 8 \
    --lr 1e-5 \
    --beta 0.1 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --max_length 1024 \
    --weight_decay 0.01 \
    --warmup_steps 60 \
    --grad_checkpoint

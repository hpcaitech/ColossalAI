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
set_n_least_used_CUDA_VISIBLE_DEVICES 8
# export CUDA_VISIBLE_DEVICES=6

PROJECT_NAME="llama2-dpo"
PARENT_SAVE_DIR="/home/yeanbang/data/experiments/dpo_ocra/ckpt"
PARENT_TENSORBOARD_DIR="/home/yeanbang/data/experiments/dpo_ocra/tensorboard"
PARENT_CONFIG_FILE="/home/yeanbang/data/experiments/dpo_ocra/train_config"
PRETRAINED_MODEL_PATH="/home/yeanbang/data/experiments/sft/SlimOrca/ckptllama2-sft-2024-01-12-11-50-56/epoch-0_step-5000/modeling"
PRETRAINED_TOKENIZER_PATH="princeton-nlp/Sheared-LLaMA-1.3B"
declare -a dataset=(
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00000
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00001
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00002
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00003
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00004
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00005
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00006
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00007
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00008
    /home/yeanbang/data/experiments/dpo_ocra/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 8 --hostfile hostfile --master_port 31312 train_dpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --checkpoint_path $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 1000 \
    --save_dir $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --max_epochs 1 \
    --accumulation_steps 4 \
    --batch_size 2 \
    --lr 5e-5 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --grad_checkpoint \
    --use_wandb

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
# NCCL IB environment variables
export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export OMP_NUM_THREADS=8


PROJECT_NAME="llama2-dpo"
PARENT_SAVE_DIR="/home/lcyab/data/models/coati_refactor_experiments/output_dpo/ckpt"
PARENT_TENSORBOARD_DIR="/home/lcyab/data/models/coati_refactor_experiments/output_dpo/tensorboard"
PARENT_CONFIG_FILE="/home/lcyab/data/models/coati_refactor_experiments/output_dpo/train_config"
PRETRAINED_MODEL_PATH="/home/lcyab/data/models/coati_refactor_experiments/sft/output/ckptllama2-sft-2023-11-28-21-10-49/epoch-0_step-5000/modeling"  #"/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"
PRETRAINED_TOKENIZER_PATH="/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"
declare -a dataset=(
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00000
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00001
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00002
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00003
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00004
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00005
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00006
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00007
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00008
    /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 8 --hostfile hostfile --master_port 30035 train_dpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --checkpoint_path $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --plugin "3d" \
    --save_interval 1000 \
    --save_dir $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --max_epochs 5 \
    --accumulation_steps 4 \
    --batch_size 4 \
    --tp 8 \
    --lr 5e-6 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --grad_checkpoint \
    --use_wandb

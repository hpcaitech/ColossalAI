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


PROJECT_NAME="llama2-ppo"
PARENT_SAVE_DIR="/home/lcyab/data/models/coati_refactor_experiments/output/ppo/ckpt"
PARENT_TENSORBOARD_DIR="/home/lcyab/data/models/coati_refactor_experiments/output/ppo/tensorboard"
PARENT_CONFIG_FILE="/home/lcyab/data/models/coati_refactor_experiments/output/ppo/train_config"
PRETRAINED_MODEL_PATH="/home/lcyab/data/models/coati_refactor_experiments/sft/output/ckptllama2-sft-2023-11-28-21-10-49/epoch-0_step-5000/modeling"  #"/home/lcyab/data/models/experiments5/checkpoint/experiment5-2023-10-20-21-53-51/modeling/"  #"/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"
REWARD_MODEL_PATH="/home/lcyab/data/models/coati_refactor_experiments/rm/output/ckptllama2-rm-2023-11-28-13-17-45/epoch-1_step-4748/modeling"  #"/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"
PRETRAINED_TOKENIZER_PATH="/home/lcyab/data/models/Sheared-LLaMA-1.3B"  # "/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"  # "/home/lcyab/data/models/bloom-560m" #
declare -a prompt_dataset=(
    # /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/arrow/part-00000
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00000
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00001
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00002
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00003
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00004
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00005
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00006
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00007
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00008
    /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow/part-00009
)

declare -a ptx_dataset=(
    /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/arrow/part-00000
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 4 --hostfile hostfile --master_port 30039 train_ppo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --rm_pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --rm_checkpoint_path $REWARD_MODEL_PATH \
    --prompt_dataset ${prompt_dataset[@]} \
    --pretrain_dataset ${ptx_dataset[@]} \
    --ptx_batch_size 1 \
    --ptx_coef 0.0 \
    --plugin "zero2" \
    --save_interval 200 \
    --save_path $SAVE_DIR \
    --num_episodes 2000 \
    --num_collect_steps 1 \
    --num_update_steps 1 \
    --experience_batch_size 8 \
    --train_batch_size 4 \
    --accumulation_steps 2 \
    --lr 9e-6 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --grad_checkpoint \
    --use_wandb

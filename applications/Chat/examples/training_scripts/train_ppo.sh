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
PARENT_SAVE_DIR="save_dir/ckpt"
PARENT_TENSORBOARD_DIR="save_dir/tensorboard"
PARENT_CONFIG_FILE="save_dir/train_config"
PRETRAINED_MODEL_PATH="sft_model_save_dir/modeling"
REWARD_MODEL_PATH="reward_model_save_dir/modeling"
PRETRAINED_TOKENIZER_PATH="pretrained/model/path"  # "/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"  # "/home/lcyab/data/models/bloom-560m" #
declare -a prompt_dataset=(
    path/to/prompt/data/arrow/part-00000
    path/to/prompt/data/arrow/part-00001
    path/to/prompt/data/arrow/part-00002
    path/to/prompt/data/arrow/part-00003
    path/to/prompt/data/arrow/part-00004
    path/to/prompt/data/arrow/part-00005
    path/to/prompt/data/arrow/part-00006
    path/to/prompt/data/arrow/part-00007
    path/to/prompt/data/arrow/part-00008
    path/to/prompt/data/arrow/part-00009
)

declare -a ptx_dataset=(
    path/to/ptx/data/arrow/part-00000
    path/to/ptx/data/arrow/part-00001
    path/to/ptx/data/arrow/part-00002
    path/to/ptx/data/arrow/part-00003
    path/to/ptx/data/arrow/part-00004
    path/to/ptx/data/arrow/part-00005
    path/to/ptx/data/arrow/part-00006
    path/to/ptx/data/arrow/part-00007
    path/to/ptx/data/arrow/part-00008
    path/to/ptx/data/arrow/part-00009
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

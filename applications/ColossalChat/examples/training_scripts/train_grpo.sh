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
set_n_least_used_CUDA_VISIBLE_DEVICES 3

PROJECT_NAME="PPO-RLVR"

PARENT_SAVE_DIR="/home/yeanbang/experiments/rlhf/grpo/model" # Path to a folder to save checkpoints
PARENT_CONFIG_FILE="/home/yeanbang/experiments/rlhf/grpo/model" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/home/yeanbang/data/model/MiniCPM-2B-128k" # local pretrained model path (from RLHF step 1: SFT)
PRETRAINED_TOKENIZER_PATH="/home/yeanbang/data/model/MiniCPM-2B-128k" # huggingface or local tokenizer path
CONVERSATION_TEMPLATE_CONFIG_PATH="/home/yeanbang/ColossalAI/applications/ColossalChat/conversation_template/MiniCPM-2b.json" # path to the conversation config file
LOGDIR="/home/yeanbang/experiments/rlhf/grpo/log"

declare -a prompt_dataset=(
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00000
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00001
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00002
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00003
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00004
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00005
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00006
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00007
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00008
    /home/yeanbang/data/dataset/tool_dataset_for_testing/tldr/data/tokenized/arrow/part-00009
)

declare -a ptx_dataset=(
    YOUR/SFT/DATA/DIR/arrow/part-00000
    YOUR/SFT/DATA/DIR/arrow/part-00001
    YOUR/SFT/DATA/DIR/arrow/part-00002
    YOUR/SFT/DATA/DIR/arrow/part-00003
    YOUR/SFT/DATA/DIR/arrow/part-00004
    YOUR/SFT/DATA/DIR/arrow/part-00005
    YOUR/SFT/DATA/DIR/arrow/part-00006
    YOUR/SFT/DATA/DIR/arrow/part-00007
    YOUR/SFT/DATA/DIR/arrow/part-00008
    YOUR/SFT/DATA/DIR/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

colossalai run --nproc_per_node 3 --hostfile hostfile --master_port 31312 train_grpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --prompt_dataset ${prompt_dataset[@]} \
    --conversation_template_config $CONVERSATION_TEMPLATE_CONFIG_PATH \
    --ptx_coef 0.0 \
    --plugin "zero2" \
    --save_interval 500 \
    --save_path $SAVE_DIR \
    --num_episodes 2000 \
    --num_collect_steps 16 \
    --num_update_steps 1 \
    --experience_batch_size 1 \
    --train_batch_size 4 \
    --accumulation_steps 4 \
    --lr 1e-6 \
    --mixed_precision "bf16" \
    --grad_clip 0.1\
    --weight_decay 0.01 \
    --kl_coef 0.5 \
    --warmup_steps 40 \
    --log_dir $LOGDIR \
    --use_flash_attn \
    --grad_checkpoint

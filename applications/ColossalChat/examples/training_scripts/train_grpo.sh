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

PROJECT_NAME="PPO-RLVR"

PARENT_SAVE_DIR="/home/yeanbang/experiments/grpo/model_extend" # Path to a folder to save checkpoints
PARENT_CONFIG_FILE="/home/yeanbang/experiments/grpo" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/home/yeanbang/experiments/grpo/sft/modelSFT-2025-02-16-18-28-49/modeling" # local pretrained model path (from RLHF step 1: SFT)
PRETRAINED_TOKENIZER_PATH="/mnt/jfs-hdd/share/models/Qwen2.5-3B" # huggingface or local tokenizer path
CONVERSATION_TEMPLATE_CONFIG_PATH="/home/yeanbang/ColossalAI/applications/ColossalChat/conversation_template/Qwen_Qwen2.5-7B-Instruct.json" # path to the conversation config file
LOGDIR="/home/yeanbang/experiments/grpo/log_model_extend"

declare -a prompt_dataset=(
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00000
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00001
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00002
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00003
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00004
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00005
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00006
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00007
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00008
    /home/yeanbang/data/competition_math/data/tokenized/prompt_new/arrow/part-00009
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

colossalai run --nproc_per_node 8 --num_nodes 1 --hostfile ./hostfile train_grpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --prompt_dataset ${prompt_dataset[@]} \
    --conversation_template_config $CONVERSATION_TEMPLATE_CONFIG_PATH \
    --ptx_coef 0.0 \
    --plugin "zero2_cpu" \
    --reward_functions math_competition_reward_fn \
    --save_interval 250 \
    --save_path $SAVE_DIR \
    --num_episodes 100 \
    --num_collect_steps 8 \
    --num_update_steps 1 \
    --experience_batch_size 1 \
    --train_batch_size 4 \
    --inference_batch_size 4 \
    --logits_forward_batch_size 1 \
    --accumulation_steps 4 \
    --lr 1e-6 \
    --mixed_precision "bf16" \
    --grad_clip 0.1\
    --weight_decay 0.01 \
    --kl_coef 0.01 \
    --warmup_steps 40 \
    --max_length 4000 \
    --max_seq_len 3700 \
    --log_dir $LOGDIR \
    --use_flash_attn \
    --grad_checkpoint

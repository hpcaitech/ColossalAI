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
set_n_least_used_CUDA_VISIBLE_DEVICES 2

# torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai_zero2

# the args satisfied: train_batch_size = num_collect_steps * experience_batch_size
torchrun --standalone --rdzv_endpoint="localhost:12355" --nproc_per_node=1 train_prompts.py \
    --pretrain_dataset /home/lcyab/data/Anthropic_rlhf/pretrain_data.json \
    --prompt_dataset /home/lcyab/data/Anthropic_rlhf/prompts_en.jsonl \
    --strategy colossalai_zero2 \
    --num_episodes 2000 --num_collect_steps 2 --num_update_steps 1 \
    --experience_batch_size 8 \
    --train_batch_size 16 \
    --save_path '/home/lcyab/data/Anthropic_rlhf/actor/v3_5' \
    --ptx_coef 0.0 \
    --rm_pretrain 'sugam11/gpt2-rlhf-reward' \
    --reward_model_tokenizer 'microsoft/DialogRPT-updown' \
    --pretrain '/home/lcyab/data/Anthropic_rlhf/actor/pretrain_v3' \
    --use_wandb


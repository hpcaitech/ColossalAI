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

# the args satisfied: train_batch_size = number_of_node_in_hostfile * num_collect_steps * experience_batch_size
# the real batch size for gradient descent is nproc_per_node * train_batch_size
colossalai run --nproc_per_node 1 --master_port 28567 --hostfile ./hostfile train_ppo.py \
    --pretrain_dataset /home/lcyab/data/Anthropic_rlhf/pretrain_data.json \
    --prompt_dataset /home/lcyab/data/Anthropic_rlhf/prompts_en.jsonl \
    --strategy colossalai_zero2 \
    --num_episodes 8000 --num_collect_steps 1 --num_update_steps 1 \
    --experience_batch_size 32 \
    --train_batch_size 32 \
    --save_path '/home/lcyab/data/Anthropic_rlhf/actor/v3_5' \
    --ptx_coef 0.0 \
    --rm_model 'gpt2' \
    --rm_pretrain 'gpt2' \
    --rm_path '/home/lcyab/data/Anthropic_rlhf/reward_model_v2_1' \
    --reward_model_tokenizer 'gpt2' \
    --pretrain '/home/lcyab/data/Anthropic_rlhf/actor/pretrain_v3' \
    --lora_rank 30 \
    --use_wandb
    # --pretrain_dataset /path/to/pretrain_data.json \
    # --prompt_dataset /path/to/prompt_dataset.jsonl \
    # --strategy colossalai_zero2 \
    # --num_episodes 8000 --num_collect_steps 1 --num_update_steps 1 \
    # --experience_batch_size 32 \
    # --train_batch_size 32 \
    # --save_path '/path/to/actor/ppo_checkpoint' \
    # --ptx_coef 0.0 \
    # --rm_model 'gpt2' \
    # --rm_pretrain 'gpt2' \
    # --rm_path '/path/to/reward_model' \
    # --reward_model_tokenizer 'gpt2' \
    # --model 'gpt2' \
    # --pretrain '/path/to/actor/pretrain_path' \
    # --use_wandb \

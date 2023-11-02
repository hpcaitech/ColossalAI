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

# the args satisfied: train_batch_size = num_collect_steps * experience_batch_size
colossalai run --nproc_per_node=1 --master_port 12354 --hostfile ./hostfile train_ppo.py \
    --pretrain_dataset /path_to/pretrain_data.json \
    --prompt_dataset /path_to/prompts_data.jsonl \
    --strategy colossalai_zero2 \
    --num_episodes 8000 --num_collect_steps 1 --num_update_steps 1 \
    --experience_batch_size 32 \
    --train_batch_size 32 \
    --save_path '/path_to/save_path' \
    --ptx_coef 0.0 \
    --rm_model 'gpt2' \
    --rm_pretrain 'gpt2' \
    --rm_path '/path_to/reward_model' \
    --reward_model_tokenizer 'gpt2' \
    --pretrain '/path_to/pretrain' \
    --use_wandb

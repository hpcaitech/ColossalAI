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

# torchrun --standalone --nproc_per_node=2 train_prompts.py prompts.csv --strategy colossalai_zero2

# the args satisfied: train_batch_size = num_collect_steps * experience_batch_size
torchrun --standalone --rdzv_endpoint="localhost:12355" --nproc_per_node=2 train_prompts_dpo.py \
    --dataset Anthropic/hh-rlhf \
    --strategy colossalai_zero2 \
    --batch_size 20 \
    --model gpt2 \
    --max_epoch 4 \
    --lr 1e-6 \
    --max_datasets_size 160000 \
    --save_path '/home/lcyab/data/Anthropic_rlhf/actor/dpo_opt_v1' \
    --pretrain '/home/lcyab/data/Anthropic_rlhf/actor/ppo_pretrain_v0' \
    --accumulation_steps 1 \
    --dataset_cache_dir '/home/lcyab/data/Anthropic_rlhf/cached_data_gpt2_full' \
    --grad_checkpoint \
    --use_wandb

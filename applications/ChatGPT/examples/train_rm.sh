set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 2

# torchrun --standalone --nproc_per_node=2 train_reward_model.py --pretrain 'bigscience/bloomz-560m' --model 'bloom' --strategy colossalai_zero2
torchrun --standalone --nproc_per_node=2 train_reward_model.py  --model 'gpt2' --strategy colossalai_zero2
# torchrun --standalone --nproc_per_node=2 train_reward_model.py --pretrain "facebook/opt-350m" --model 'opt' --strategy colossalai_zero2

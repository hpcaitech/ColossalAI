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

torchrun --standalone --nproc_per_node=2 train_prompts.py \
    --model "gpt2" \
    --pretrain "/path/to/sft_gpt2" \
    --rm_model "gpt2" \
    --rm_path "/path/to/gpt2_rm" \
    --pretrain_dataset /path/to/pretrain_data.json \
    --prompt_dataset /path/to/prompt_data.json \
    --strategy colossalai_zero2 \
    --num_episodes 10 \
    --num_collect_steps 2 \
    --num_update_steps 2 \
    --train_batch_size 4 \
    --experience_batch_size 8 \
    --ptx_batch_size 4 \
    --max_input_len 96 \
    --max_seq_len 256 \
    --seq_chunk_size 8

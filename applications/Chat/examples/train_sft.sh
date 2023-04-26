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

set_n_least_used_CUDA_VISIBLE_DEVICES 8
CUDA_VISIBLE_DEVICES=4,5,6,7
#torchrun --standalone --nproc_per_node=2 train_sft.py --pretrain 'bigscience/bloomz-560m' --model 'bloom' --strategy colossalai_zero2 --log_interval 10
#torchrun --standalone --nproc_per_node=8 train_sft.py  --model 'gpt2' --strategy colossalai_zero2 --batch_size 1 --log_interval 10
torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain "/data/scratch/LLaMa-7B" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --log_interval 10 \
    --save_path /data3/users/lczht \
    --dataset /home/lczht/data2/Coati/examples/alpaca_data.json \
    --batch_size 1 \
    --max_datasets_size 128 \
    --lora_rank 4

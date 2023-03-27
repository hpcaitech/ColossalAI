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

#torchrun --standalone --nproc_per_node=2 train_sft.py --pretrain 'bigscience/bloomz-560m' --model 'bloom' --strategy colossalai_zero2 --log_interval 10
#torchrun --standalone --nproc_per_node=8 train_sft.py  --model 'gpt2' --strategy colossalai_zero2 --batch_size 1 --log_interval 10
torchrun --standalone --nproc_per_node=8 train_sft.py \
    --pretrain "/data/personal/nus-mql/LLAMA-7B" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --log_interval 10 \
    --save_path /data/personal/nus-mql/Coati-7B \
    --dataset /data/personal/nus-mql/stanford_alpaca/alpaca_data.json

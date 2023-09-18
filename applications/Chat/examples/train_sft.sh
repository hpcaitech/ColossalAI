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

torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain 'bigscience/bloom-560m' \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --log_interval 10 \
    --tensorboard_dir "/root/test-coati/ColossalAI/applications/Chat/examples/tensorboard" \
    --save_path "/root/test-coati/ColossalAI/applications/Chat/examples/output" \
    --dataset "/root/test-coati/ColossalAI/applications/Chat/examples/data.json" \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1

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
set_n_least_used_CUDA_VISIBLE_DEVICES 4

PROJECT_NAME="simpo"
PARENT_CONFIG_FILE="./benchmark_config" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="" # huggingface or local tokenizer path
BENCHMARK_DATA_DIR="./temp/simpo" # Path to benchmark data
DATASET_SIZE=640

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
declare -a dataset=(
    $BENCHMARK_DATA_DIR/arrow/part-0
)

# Generate dummy test data
python prepare_dummy_test_dataset.py --data_dir $BENCHMARK_DATA_DIR --dataset_size $DATASET_SIZE --max_length 2048 --data_type preference


colossalai run --nproc_per_node 4 --master_port 31313 ../examples/training_scripts/train_dpo.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2_cpu" \
    --loss_type "simpo_loss" \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --batch_size 8 \
    --lr 1e-6 \
    --beta 0.1 \
    --gamma 0.6 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --max_length 2048 \
    --weight_decay 0.01 \
    --warmup_steps 60 \
    --disable_reference_model \
    --length_normalization \
    --grad_checkpoint \
    --use_flash_attn

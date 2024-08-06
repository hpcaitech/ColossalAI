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

PROJECT_NAME="sft"
PARENT_CONFIG_FILE="./benchmark_config" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="" # huggingface or local tokenizer path
BENCHMARK_DATA_DIR="./temp/sft" # Path to benchmark data
DATASET_SIZE=640

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"
declare -a dataset=(
    $BENCHMARK_DATA_DIR/arrow/part-0
)


# Generate dummy test data
python prepare_dummy_test_dataset.py --data_dir $BENCHMARK_DATA_DIR --dataset_size $DATASET_SIZE --max_length 2048 --data_type sft


# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
colossalai run --nproc_per_node 1 --master_port 31312 ../examples/training_scripts/train_sft.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --plugin zero2 \
    --batch_size 8 \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --lr 5e-5 \
    --lora_rank 32 \
    --max_len 2048 \
    --grad_checkpoint \
    --use_flash_attn

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
PROJECT_NAME="SFT"
PARENT_SAVE_DIR="" # Path to a folder to save checkpoints
PARENT_CONFIG_FILE="" # Path to a folder to save training config logs
PARENT_LOG_DIR="" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="" # huggingface or local tokenizer path
declare -a dataset=(
    YOUR/SFT/DATA/DIR/arrow/part-00000
    YOUR/SFT/DATA/DIR/arrow/part-00001
    YOUR/SFT/DATA/DIR/arrow/part-00002
    YOUR/SFT/DATA/DIR/arrow/part-00003
    YOUR/SFT/DATA/DIR/arrow/part-00004
    YOUR/SFT/DATA/DIR/arrow/part-00005
    YOUR/SFT/DATA/DIR/arrow/part-00006
    YOUR/SFT/DATA/DIR/arrow/part-00007
    YOUR/SFT/DATA/DIR/arrow/part-00008
    YOUR/SFT/DATA/DIR/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"
LOG_DIR="${PARENT_LOG_DIR}${FULL_PROJECT_NAME}"

echo $(which colossalai)
echo $(which python)
# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
colossalai run --nproc_per_node 4 --master_port 31312 --hostfile ./hostfile train_sft.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --save_interval 2000 \
    --dataset ${dataset[@]} \
    --plugin zero2 \
    --batch_size 8 \
    --max_epochs 1 \
    --accumulation_steps 1 \
    --lr 5e-5 \
    --max_len 4096 \
    --use_flash_attn \
    --grad_checkpoint \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --log_dir $LOG_DIR \

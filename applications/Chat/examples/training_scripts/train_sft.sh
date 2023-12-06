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
PROJECT_NAME="llama2-sft"
PARENT_SAVE_DIR="save_dir/ckpt"
PARENT_TENSORBOARD_DIR="save_dir/tensorboard"
PARENT_CONFIG_FILE="save_dir/train_config"
PRETRAINED_MODEL_PATH="pretrained/model/path"
PRETRAINED_TOKENIZER_PATH="pretrained/model/path"
declare -a dataset=(
    path/to/sft/data/arrow/part-00000
    path/to/sft/data/arrow/part-00001
    path/to/sft/data/arrow/part-00002
    path/to/sft/data/arrow/part-00003
    path/to/sft/data/arrow/part-00004
    path/to/sft/data/arrow/part-00005
    path/to/sft/data/arrow/part-00006
    path/to/sft/data/arrow/part-00007
    path/to/sft/data/arrow/part-00008
    path/to/sft/data/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
colossalai run --nproc_per_node 3 --master_port 28534 --hostfile ./hostfile train_sft.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --dataset ${dataset[@]} \
    --save_interval 5000 \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --lora_rank 32 \
    --plugin zero2 \
    --batch_size 2 \
    --max_epochs 1 \
    --accumulation_steps 4 \
    --lr 2e-5 \
    --max_len 2048 \
    --use_wandb

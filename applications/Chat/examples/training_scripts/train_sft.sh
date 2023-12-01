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
# set_n_least_used_CUDA_VISIBLE_DEVICES 3
export CUDA_VISIBLE_DEVICES=1,2,3
PROJECT_NAME="llama2-sft"
PARENT_SAVE_DIR="/home/lcyab/data/models/coati_refactor_experiments/sft_lora/output/ckpt"
PARENT_TENSORBOARD_DIR="/home/lcyab/data/models/coati_refactor_experiments/sft_lora/output/tensorboard"
PARENT_CONFIG_FILE="/home/lcyab/data/models/coati_refactor_experiments/sft_lora/output/train_config"
PRETRAINED_MODEL_PATH="/home/lcyab/data/models/Sheared-LLaMA-1.3B" #"/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"
PRETRAINED_TOKENIZER_PATH="/home/lcyab/data/models/Sheared-LLaMA-1.3B" #"/mnt/vepfs/lcxyc/leaderboard_models/Colossal-LLaMA-2-7b-base/"
declare -a dataset=(
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00000
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00001
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00002
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00003
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00004
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00005
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00006
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00007
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00008
    /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow/part-00009
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

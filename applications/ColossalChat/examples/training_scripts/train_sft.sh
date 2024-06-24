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


# export CUDA_VISIBLE_DEVICES=4,5,6
set_n_least_used_CUDA_VISIBLE_DEVICES 2
PROJECT_NAME="sft"
PARENT_SAVE_DIR="/home/yeanbang/data/experiment/rlhf_cont/dpo/ckpt" # Path to a folder to save checkpoints
PARENT_TENSORBOARD_DIR="/home/yeanbang/data/experiment/rlhf_cont/dpo/log" # Path to a folder to save logs
PARENT_CONFIG_FILE="/home/yeanbang/data/experiment/rlhf_cont/dpo/log" # Path to a folder to save training config logs
PRETRAINED_MODEL_PATH="/home/yeanbang/data/models/Sheared-LLaMA-1.3B" # huggingface or local model path
PRETRAINED_TOKENIZER_PATH="/home/yeanbang/data/models/Sheared-LLaMA-1.3B" # huggingface or local tokenizer path
declare -a dataset=(
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00000
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00001
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00002
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00003
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00004
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00005
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00006
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00007
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00008
    /home/yeanbang/data/experiment/rlhf_cont/dpo/dataset_tokenized/sft/arrow/part-00009
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}-${FULL_PROJECT_NAME}.json"

echo $(which colossalai)
echo $(which python)
# the real batch size for gradient descent is number_of_node_in_hostfile * nproc_per_node * train_batch_size
colossalai run --nproc_per_node 1 --master_port 31312 --hostfile ./hostfile train_sft.py \
    --pretrain $PRETRAINED_MODEL_PATH \
    --tokenizer_dir $PRETRAINED_TOKENIZER_PATH \
    --save_interval 4000 \
    --dataset ${dataset[@]} \
    --save_path $SAVE_DIR \
    --config_file $CONFIG_FILE \
    --lora_rank 0 \
    --plugin zero2 \
    --batch_size 4 \
    --max_epochs 1 \
    --accumulation_steps 4 \
    --lr 5e-5 \
    --max_len 1000 \
    --grad_checkpoint \
    --use_wandb \
    --use_flash_attn

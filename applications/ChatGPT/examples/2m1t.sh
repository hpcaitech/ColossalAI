# set_n_least_used_CUDA_VISIBLE_DEVICES() {
#     local n=${1:-"9999"}
#     echo "GPU Memory Usage:"
#     local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
#         | tail -n +2 \
#         | nl -v 0 \
#         | tee /dev/tty \
#         | sort -g -k 2 \
#         | awk '{print $1}' \
#         | head -n $n)
#     export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
#     echo "Now CUDA_VISIBLE_DEVICES is set to:"
#     echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# }
# 
# set_n_least_used_CUDA_VISIBLE_DEVICES 2

export CUDA_VISIBLE_DEVICES="0,1,2"

torchrun --standalone 2m1t.py "./awesome-chatgpt-prompts/prompts.csv" \
    --strategy naive --lora_rank 2 --pretrain "facebook/opt-350m" --model 'opt' --debug

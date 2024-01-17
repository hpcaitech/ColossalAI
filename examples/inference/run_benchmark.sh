ROOT=$(realpath $(dirname $0))
PY_SCRIPT=${ROOT}/benchmark_llama.py
GPU=$(nvidia-smi -L | head -1 | cut -d' ' -f4 | cut -d'-' -f1)
mode=$1

mkdir -p logs

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage() {
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

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 1

# benchmark llama2-7b one single GPU
for bsz in 16 32 64; do
    python3 ${PY_SCRIPT} -m llama2-7b --tp_size 1 --pp_size 1 -b $bsz -s 256 --output_len 128 --mode ${mode} | tee logs/${mode}_${GPU}_${bsz}_256.txt
done


for bsz in 16 32 64; do
    python3 ${PY_SCRIPT} -m llama2-7b --tp_size 1 --pp_size 1 -b $bsz -s 1024 --output_len 128 --mode ${mode} | tee logs/${mode}_${GPU}_${bsz}_1024.txt
done

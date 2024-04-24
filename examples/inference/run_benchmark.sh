ROOT=$(realpath $(dirname $0))
echo $ROOT
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
for input_len in  128 512 1024; do
    for output_len in 128 256; do
        for bsz in 16 32 64; do
            python3 ${PY_SCRIPT} -m llama2-7b --tp_size 1 -b ${bsz} -s ${input_len} --output_len ${output_len} --mode ${mode} --test_random_weight | tee logs/${bsz}_${input_len}_${output_len}_${mode}_${GPU}.txt
        done
    done
done

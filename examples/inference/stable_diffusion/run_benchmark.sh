#!/bin/bash

models=("PixArt-alpha/PixArt-XL-2-1024-MS" "stabilityai/stable-diffusion-3-medium-diffusers")
parallelism=(1 2 4 8)
resolutions=(1024 2048 3840)
modes=("colossalai" "diffusers")

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

for model in "${models[@]}"; do
    for p in "${parallelism[@]}"; do
        for resolution in "${resolutions[@]}"; do
            for mode in "${modes[@]}"; do
                if [[ "$mode" == "colossalai" && "$p" == 1 ]]; then
                    continue
                fi
                if [[ "$mode" == "diffusers" && "$p" != 1 ]]; then
                    continue
                fi
                CUDA_VISIBLE_DEVICES_set_n_least_memory_usage $p

                cmd="python examples/inference/stable_diffusion/benchmark_sd3.py -m \"$model\" -p $p --mode $mode --log -H $resolution -w $resolution"

                echo "Executing: $cmd"
                eval $cmd
            done
        done
    done
done

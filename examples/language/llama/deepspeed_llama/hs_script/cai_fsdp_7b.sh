#!/bin/bash

module load cuda/11.7.1-gcc-9.4.0
module load nccl/2.14.3-1-gcc-9.4.0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3//etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /mnt/vepfs/conda/envs/llama_2

ROOT=$(pwd)

cd ../..

colossalai run --nproc_per_node 8 --hostfile ${ROOT}/cai_host_4.txt --master_addr 192.168.0.189 benchmark.py --plugin "fsdp" -l 512 -g
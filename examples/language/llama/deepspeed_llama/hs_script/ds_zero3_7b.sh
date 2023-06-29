#!/bin/bash

module load cuda/11.7.1-gcc-9.4.0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/vepfs/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/vepfs/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/vepfs/miniconda3//etc/profile.d/conda.sh"
    else
        export PATH="/mnt/vepfs/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate llama

ROOT=$(pwd)

cd ..

deepspeed --num_nodes 2 --num_gpus 16 --master_addr 180.184.78.121 --master_port 29503 --hostfile=${ROOT}/host_file_2.txt \
	ds_benchmark.py -l 512 -g \
	--deepspeed --deepspeed_config ${ROOT}/zero3.json
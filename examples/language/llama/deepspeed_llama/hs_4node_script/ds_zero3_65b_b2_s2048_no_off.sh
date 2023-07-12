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

export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

cd ..

deepspeed --master_addr 192.168.0.189 --master_port 29503 --hostfile=${ROOT}/host_file_4.txt \
	ds_benchmark.py -l 2048 -w 32 -c '65b' -train_micro_batch_size_per_gpu 2 -g \
	--deepspeed --deepspeed_config ${ROOT}/zero3_no_off.json > ${ROOT}/ds_zero3_65b_b2_s2048_no_off.log 2>&1
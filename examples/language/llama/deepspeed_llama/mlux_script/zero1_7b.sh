#!/bin/bash

module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load GCC/11.3.0

hostid_start=3
if [ $SLURM_NNODES -gt 1 ]
then
    hostid_start=$(( $hostid_start + 1 ))
fi
master="mel${SLURM_NODELIST:hostid_start:4}"
master=`host $master| grep address | awk '{print $4}'`

export MASTER_ADDR=$master
export MASTER_PORT=29500
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_RANK=0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/users/u100034/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/users/u100034/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/users/u100034/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/users/u100034/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate llama

ROOT=$(pwd)
cd /mnt/tier2/users/u100034/ColossalAI_llama/examples/language/llama/deepspeed_llama
python -u \
        ds_benchmark.py -l 512 \
        --deepspeed --deepspeed_config ${ROOT}/zero1.json --world_size $WORLD_SIZE --local_rank 0
#!/bin/bash -l
#SBATCH --job-name=llama_pretrain
#SBATCH --account=p200164 # project account
#SBATCH --partition=gpu                    # partition
#SBATCH --nodes=2                          # number of nodes
#SBATCH --ntasks=2                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --cpus-per-task=32                  # number of cores per task
#SBATCH --gpus-per-task=4                  # number of gpu per task
#SBATCH --time=00-01:00                     # time (DD-HH:MM)
#SBATCH --qos=default
#SBATCH --contiguous



export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=0

export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=test

srun ./zero1_7b.sh --o out1.log
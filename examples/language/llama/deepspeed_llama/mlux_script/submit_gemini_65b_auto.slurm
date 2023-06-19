#!/bin/bash -l
#SBATCH --job-name=llama_pretrain
#SBATCH --account=p200164 # project account
#SBATCH --partition=gpu                    # partition
#SBATCH --nodes=16                          # number of nodes
#SBATCH --ntasks=64                         # number of tasks
#SBATCH --ntasks-per-node=4                # number of tasks per node
#SBATCH --cpus-per-task=128                  # number of cores per task
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --time=0-00:20                     # time (DD-HH:MM)
#SBATCH --qos=default
#SBATCH --contiguous



export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=0

export OMP_NUM_THREADS=4
export PYTHONNOUSERSITE=test

srun ./gemini_65b_auto.sh
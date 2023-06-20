#!/bin/bash


hostid_start=3
if [ $SLURM_NNODES -gt 1 ]
then
    hostid_start=$(( $hostid_start + 1 ))
fi
master="mel${SLURM_NODELIST:hostid_start:4}"
master=`host $master| grep address | awk '{print $4}'`

export MASTER_ADDR=$master
export MASTER_PORT=13245
export LOCAL_SIZE=4
rank=$SLURM_PROCID
nprocs_per_node=4

# WORLD_SIZE=$SLURM_NPROCS
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

nodes_ip=`scontrol show hostnames $SLURM_JOB_NODELIST`
# shellcheck disable=SC2068
local_node=$SLURM_NODEID
echo $local_node
if $local_node -eq 0;
then
   # shellcheck disable=SC2068
   for var in ${nodes_ip[@]}
      do
         echo "${var} slots=4 " >> nodes_ip.txt
         echo $var
      done
fi



#DISTRIBUTED_ARGS="--nproc_per_node $nprocs_per_node \
#                  --nnodes $SLURM_NNODES \
#                  --node_rank ${rank} \
#                  --master_addr ${MASTER_ADDR} \
#                  --master_port ${MASTER_PORT}"

cd ..

deepspeed --num_nodes 2 --num_gpus 8 --hostfile mlux_script/nodes_ip.txt \
	ds_benchmark.py -l 512 \
	--deepspeed --deepspeed_config zero.json





#!/usr/bin/env sh
usage()
{
    echo "Usage: $0 [-p PORT] python_file [config_file]"
    exit 1
}

while getopts "p:h" OPTION
do
    case ${OPTION} in
        p) port=${OPTARG} ;;
	h) usage $0 ;;
	?) usage $0 ;;
    esac
done

shift "$((OPTIND-1))"

world_size=$SLURM_NPROCS
rank=$SLURM_PROCID
nprocs_per_node=$(( $SLURM_NPROCS / $SLURM_NNODES ))
local_rank=$(( $SLURM_PROCID % $nprocs_per_node ))

hostid_start=3
if [ $SLURM_NNODES -gt 1 ]
then
    hostid_start=$(( $hostid_start + 1 ))
fi
master="mel${SLURM_NODELIST:hostid_start:4}"
master=`host $master| grep address | awk '{print $4}'`
port=${port:-29500}

#echo "#nodes=$SLURM_NNODES node_id=$SLURM_NODEID world_size=$SLURM_NPROCS rank=$SLURM_PROCID local_rank=$local_rank master=$master:$port"

python_file=$1
config_file=$2
if [ ! "$python_file" ]
then
    usage $0
fi

#echo "$python_file $config_file"
export LOCAL_RANK=0
export MASTER_PORT=$port
export MASTER_ADDR=$master
export WORLD_SIZE=$world_size
export RANK=$rank

#export TORCH_EXTENSIONS_DIR=$MRPJ/torch_jit
unset OMPI_COMM_WORLD_LOCAL_RANK
export DATA=/project/scratch/p200012/dataset/imagenet-100


if [ "$config_file" ]
then
    NCCL_ALGO=Ring NCCL_CROSS_NIC=1 python -u $python_file --host=$master --config=$config_file
else
    NCCL_ALGO=Ring NCCL_CROSS_NIC=1 python -u $python_file --host=$master
fi

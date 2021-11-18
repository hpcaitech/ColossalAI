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

if [ "$config_file" ]
then
    CUDA_VISIBLE_DEVICES=$local_rank NCCL_CROSS_NIC=1 python $python_file --local_rank=$rank --world_size=$world_size --host=$master --port=$port --config=$config_file
else
    CUDA_VISIBLE_DEVICES=$local_rank NCCL_CROSS_NIC=1 python $python_file --local_rank=$rank --world_size=$world_size --host=$master --port=$port
fi

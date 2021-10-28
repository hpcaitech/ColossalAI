#!/usr/bin/env sh


main_file=$1
config_file=$2

python $main_file --local_rank $SLURM_PROCID --world_size $SLURM_NPROCS --host $HOST --port 29500 --config $config_file

# how to run this script
# exmaple:
# HOST=IP_ADDR srun ./scripts/slurm_dist_train.sh ./examples/train_vit_2d.py ./configs/vit/vit_2d.py
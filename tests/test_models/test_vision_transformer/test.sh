#!/usr/bin/env sh
test_file=$1

python $test_file --local_rank $SLURM_PROCID --world_size $SLURM_NPROCS --host $HOST --port 29500

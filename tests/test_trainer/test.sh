#!/usr/bin/env sh
test_file=$1

python $test_file --rank $SLURM_PROCID --world_size $SLURM_NPROCS --host $HOST --port 29500

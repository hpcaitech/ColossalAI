#!/bin/bash

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts.txt)

cd ../..

export OMP_NUM_THREADS=8

colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -c 70b -g -x -b 2

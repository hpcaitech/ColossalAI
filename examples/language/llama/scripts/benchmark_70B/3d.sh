#!/bin/bash

# TODO: fix this
echo "3D parallel for LLaMA-2 is not ready yet"
exit 1

################
#Load your environments and modules here
################

HOSTFILE=$(realpath hosts.txt)

cd ../..

export OMP_NUM_THREADS=8

colossalai run --nproc_per_node 8 --hostfile $HOSTFILE benchmark.py -c 70b -p 3d -g -x -b 8 --tp 4 --pp 2 --mbs 1

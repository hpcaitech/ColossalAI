#!/bin/bash

################
#Load your environments and modules here
################

export OMP_NUM_THREADS=8

#hybird: zero2+flash_atten+grad_ckpt+bs4
colossalai run --nproc_per_node 8 benchmark.py -m "/home/grpo/models/Qwen2.5-7B/" -p "3d" -x -g --zero 1 -b 32 --mbs 1 --tp 2 --pp 2 -l 4096

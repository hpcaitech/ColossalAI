#!/bin/bash

################
#Load your environments and modules here
################

export OMP_NUM_THREADS=8

colossalai run --nproc_per_node 8 benchmark.py \
	--model_path "/home/grpo/models/DeepSeek-R1-Distill-Qwen-7B/" \
	-p "3d" \
	-x -g \
	--zero 1 \
	--cpu_offload \
	-b 16 --mbs 1 \
	--tp 4 --pp 2 \
	-l 4096 \
	-s 3 \
	&>qwen2_7b.log &

#!/bin/bash

export MASTER_ADDR=192.168.101.21
export MASTER_PORT=11455
export NODE_RANK=0

mpirun -np 4 python runcai.py with data_root=/work/zhangyq/vilt_data/arrow_coco num_gpus=1 num_nodes=1 task_mlm_itm_s whole_word_masking=True step200k per_gpu_batchsize=96

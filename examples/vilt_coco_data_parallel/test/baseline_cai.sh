#!/bin/bash

export MASTER_ADDR=$(hostname)
export MASTER_PORT=11455
export NODE_RANK=0

work_dir=/work/zhangyq/ColossalAI/examples/vilt
cd $work_dir
mpirun -np 2 python runcai.py with data_root=/work/zhangyq/vilt_data/arrow_coco_mini num_gpus=1 num_nodes=1 task_mlm_itm_s whole_word_masking=True step200k per_gpu_batchsize=96

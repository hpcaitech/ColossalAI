#!/bin/bash

set -xue
echo "this test is outdated"
# pip install -r requirements.txt

# BS=4
# MEMCAP=0
# GPUNUM=4
# MODLE="facebook/opt-125m"

# torchrun \
#   --nproc_per_node ${GPUNUM} \
#   --master_port 19198 \
#   run_clm.py \
#   -s \
#   --output_dir $PWD \
#   --mem_cap ${MEMCAP} \
#   --model_name_or_path ${MODLE} \
#   --per_device_train_batch_size ${BS} \
#   --num_train_epochs 1

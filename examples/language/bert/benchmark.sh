#!/bin/bash
set -xe

pip install -r requirements.txt

for plugin in "torch_ddp" "torch_ddp_fp16" "gemini" "low_level_zero"; do
   torchrun --standalone --nproc_per_node 2  benchmark.py --plugin $plugin --model_type "bert"
   torchrun --standalone --nproc_per_node 2  benchmark.py  --plugin $plugin --model_type "albert"
done

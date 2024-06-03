#!/bin/bash
set -xe

pip install -r requirements.txt

for plugin in "torch_ddp" "torch_ddp_fp16" "gemini" "low_level_zero"; do
    torchrun --standalone --nproc_per_node 4  finetune.py --target_f1 0.80 --plugin $plugin
done

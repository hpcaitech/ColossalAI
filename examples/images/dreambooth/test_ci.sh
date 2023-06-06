#!/bin/bash
set -xe
pip install -r requirements.txt

HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

for plugin in "torch_ddp" "torch_ddp_fp16" "gemini" "low_level_zero"; do
  torchrun --nproc_per_node 4 --standalone train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path="Your Pretrained Model Path"  \
  --instance_data_dir="Your Input Pics Path" \
  --output_dir="path-to-save-model" \
  --instance_prompt="your prompt" \
  --resolution=512 \
  --plugin=$plugin \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --placement="cuda"
done

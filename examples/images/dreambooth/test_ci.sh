#!/bin/bash
set -xe
echo "this test is slow"

# pip install -r requirements.txt

# HF_DATASETS_OFFLINE=1
# TRANSFORMERS_OFFLINE=1
# DIFFUSERS_OFFLINE=1

# #  "torch_ddp" "torch_ddp_fp16" "low_level_zero"
# for plugin in "gemini"; do
#   torchrun --nproc_per_node 4 --standalone train_dreambooth_colossalai.py \
#   --pretrained_model_name_or_path="/data/dreambooth/diffuser/stable-diffusion-v1-4"  \
#   --instance_data_dir="/data/dreambooth/Teyvat/data" \
#   --output_dir="./weight_output" \
#   --instance_prompt="a picture of a dog" \
#   --resolution=512 \
#   --plugin=$plugin \
#   --train_batch_size=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --test_run=True \
#   --num_class_images=200
# don

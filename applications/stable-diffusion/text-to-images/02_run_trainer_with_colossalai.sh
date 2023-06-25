#!/bin/bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"

torchrun --nproc_per_node 4 stable_diffusion_colossalai_trainer.py \
    --mixed_precision="fp16" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$dataset_name \
    --use_ema \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=15000 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="sd-pokemon-model" \
    --plugin="gemini" \
    --placement="cuda"

#!/bin/bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_ID="fusing/instructpix2pix-1000-samples"

torchrun --nproc_per_node 4 stable_diffusion_colossalai_trainer.py \
    --mixed_precision="fp16" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --resolution=256 --random_flip \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=12000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --plugin="gemini" \
    --placement="cuda" \
    --task_type="image_to_image" \
    --output_dir="instruct_pix2pix" 
#!/bin/bash

python inference.py --validation_prompts "comparison between beijing and shanghai" \
                                   --unet_saved_path ./sd-pokemon-model/diffusion_pytorch_model.bin \
                                   --model_id "CompVis/stable-diffusion-v1-4" \
                                   --task_type "text_to_image" \

python inference.py --validation_prompts "turn the color of the mountain into yellow" \
                                   --val_image_url https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png \
                                   --unet_saved_path ./instruct_pix2pix/diffusion_pytorch_model.bin \
                                   --model_id "CompVis/stable-diffusion-v1-4" \
                                   --task_type "image_to_image" \
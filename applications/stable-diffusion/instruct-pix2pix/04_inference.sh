#!/bin/bash

python image_to_image_inference.py --validation_prompts "the picture has some sunshine" \
                                   --val_image_url https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png \
                                   --unet_saved_path /home/lclcq/ColossalAI/applications/stable-diffusion/instruct-pix2pix/instruct-pix2pix-model/diffusion_pytorch_model.bin
                                   
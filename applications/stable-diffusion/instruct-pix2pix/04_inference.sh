#!/bin/bash

python image_to_image_inference.py --validation_prompts "turn the color of the mountains into yellow" \
                                   --val_image_url https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png \
                                   --unet_saved_path /home/lclcq/ColossalAI/applications/stable-diffusion/instruct-pix2pix/instruct-pix2pix-model/diffusion_pytorch_model.bin
                                   
#!/bin/bash
set -euxo pipefail

conda env create -f environment.yaml

conda activate ldm

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install transformers diffusers invisible-watermark

BUILD_EXT=1  pip install colossalai

wget https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt

python main.py --logdir /tmp --train --base configs/Teyvat/train_colossalai_teyvat.yaml --ckpt 512-base-ema.ckpt

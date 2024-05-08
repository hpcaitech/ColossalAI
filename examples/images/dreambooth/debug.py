"""
torchrun --standalone --nproc_per_node=1 debug.py
"""

from diffusers import AutoencoderKL

import colossalai
from colossalai.zero import ColoInitContext

path = "/data/scratch/diffuser/stable-diffusion-v1-4"

colossalai.launch_from_torch()
with ColoInitContext(device="cpu"):
    vae = AutoencoderKL.from_pretrained(
        path,
        subfolder="vae",
        revision=None,
    )

for n, p in vae.named_parameters():
    print(n)

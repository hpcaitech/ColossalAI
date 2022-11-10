import os
import sys
from copy import deepcopy

import yaml
from datetime import datetime

from diffusers import StableDiffusionPipeline
import torch
from ldm.util import instantiate_from_config
from main import get_parser

if __name__ == "__main__":
    with torch.no_grad():
        yaml_path = "../../train_colossalai.yaml"
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = f.read()
        base_config = yaml.load(config, Loader=yaml.FullLoader)
        unet_config = base_config['model']['params']['unet_config']
        diffusion_model = instantiate_from_config(unet_config).to("cuda:0")

        pipe = StableDiffusionPipeline.from_pretrained(
            "/data/scratch/diffuser/stable-diffusion-v1-4"
        ).to("cuda:0")
        dif_model_2 = pipe.unet

        random_input_ = torch.rand((4, 4, 32, 32)).to("cuda:0")
        random_input_2 = torch.clone(random_input_).to("cuda:0")
        time_stamp = torch.randint(20, (4,)).to("cuda:0")
        time_stamp2 = torch.clone(time_stamp).to("cuda:0")
        context_ = torch.rand((4, 77, 768)).to("cuda:0")
        context_2 = torch.clone(context_).to("cuda:0")

        out_1 = diffusion_model(random_input_, time_stamp, context_)
        out_2 = dif_model_2(random_input_2, time_stamp2, context_2)
        print(out_1.shape)
        print(out_2['sample'].shape)
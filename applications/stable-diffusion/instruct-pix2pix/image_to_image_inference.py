import argparse
import PIL
import requests

import torch
from torch import nn
from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from colossalai.booster import Booster

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def main():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--unet_saved_path", 
        type=str, 
        default=None, 
        help=("path of your trained unet model")
    )
    parser.add_argument(
        "--val_image_url", 
        type=str, 
        default = None, 
        help=("the url of your test image")
    )
    args = parser.parse_args()

    assert args.val_image_url is not None, "the image url has to be provided"

    model_id = "CompVis/stable-diffusion-v1-4"
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=None
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", revision=None
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", revision=None
    )

    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    if args.unet_saved_path is not None:
        print("loading trained model from {}".format(args.unet_saved_path))
        unet.load_state_dict(torch.load(args.unet_saved_path))

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )

    piplinee = pipeline.to("cuda")
    pipeline.enable_attention_slicing()

    prompt = "a photo of an astronaut riding a horse on Mars"
    if args.validation_prompts is not None:
        prompt = args.validation_prompts


    original_image = download_image(args.val_image_url)

    image = pipeline(prompt, image=original_image, num_inference_steps=20,
                         image_guidance_scale=1.5,
                         guidance_scale=7,
                        ).images[0]
        
    image.save("stable_diffusion_example_colossalai.png")


if __name__ == "__main__":
    main()


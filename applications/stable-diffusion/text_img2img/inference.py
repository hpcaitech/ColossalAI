import argparse
import PIL
import requests

import torch
from torch import nn
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, StableDiffusionInstructPix2PixPipeline
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
        "--model_id",
        type=str,
        default=None,
        help=("your trained model id or model name")
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument("--unet_saved_path", type=str, default=None, help=("path of your trained unet model"))
    parser.add_argument(
        "--val_image_url", 
        type=str, 
        default = None, 
        help=("the url of your test image")
    )
    parser.add_argument('--task_type',
                        type=str,
                        default='text_to_image',
                        choices=['text_to_image', 'image_to_image'],
                        help="plugin to use")

    args = parser.parse_args()

    model_id = args.model_id
    assert args.validation_prompts is not None, "have to provide a prompt for this inference file"
    if args.task_type == "image_to_image":
        assert args.val_image_url is not None, "the image url has to be provided"

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

    if args.task_type == "image_to_image":
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

    if args.task_type == "text_to_image":
        print("use text_to_image pipeline")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
    else:
        print("use image_to_image pipeline")
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )


    pipe = pipeline.to("cuda")
    pipe.enable_attention_slicing()
    prompt = args.validation_prompts
    if args.task_type == "text_to_image":
        print("get result from text_to_image model ...")
        image = pipe(prompt).images[0]  
        image.save("text_to_image_example.png")
    else:
        print("get result from image_to_image model ...")
        original_image = download_image(args.val_image_url)
        original_image.save("original_image.png")
        image = pipeline(prompt, image=original_image, num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            ).images[0]
            
        image.save("image_to_image_example.png")



if __name__ == "__main__":
    main()


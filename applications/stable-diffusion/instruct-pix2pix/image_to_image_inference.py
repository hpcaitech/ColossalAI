import argparse
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from colossalai.booster import Booster

def main():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument("--unet_saved_path", type=str, default=None, help=("path of your trained unet model"))
    args = parser.parse_args()

    model_id = "CompVis/stable-diffusion-v1-4"

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

    if args.unet_saved_path is not None:
        print("loading trained model from {}".format(args.unet_saved_path))
        unet.load_state_dict(torch.load(args.unet_saved_path))

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )

    pipe = pipeline.to("cuda")
    pipe.enable_attention_slicing()

    prompt = "a photo of an astronaut riding a horse on Mars"
    if args.validation_prompts is not None:
        prompt = args.validation_prompts

    image = pipe(prompt).images[0]  
        
    image.save("stable_diffusion_example_colossalai.png")


if __name__ == "__main__":
    main()


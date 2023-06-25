import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

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

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
)

pipe = pipeline.to("cuda")
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on moon"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse_colossalai.png")


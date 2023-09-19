from functools import partial

import diffusers
import torch
import transformers

from ..registry import model_zoo

BATCH_SIZE = 2
SEQ_LENGTH = 5
HEIGHT = 224
WIDTH = 224
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 7, WIDTH // 7)
TIME_STEP = 3

data_vae_fn = lambda: dict(sample=torch.randn(2, 3, 32, 32))
data_unet_fn = lambda: dict(sample=torch.randn(2, 3, 32, 32), timestep=3)

identity_output = lambda x: x
clip_vision_model_output = lambda x: dict(pooler_output=x[1])


def data_clip_model():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    position_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    pixel_values = torch.zeros((BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    return dict(
        input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids
    )


def data_clip_text():
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_clip_vision():
    pixel_values = torch.zeros((BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    return dict(pixel_values=pixel_values)


model_zoo.register(
    name="diffusers_auto_encoder_kl",
    model_fn=diffusers.AutoencoderKL,
    data_gen_fn=data_vae_fn,
    output_transform_fn=identity_output,
)

model_zoo.register(
    name="diffusers_vq_model", model_fn=diffusers.VQModel, data_gen_fn=data_vae_fn, output_transform_fn=identity_output
)

model_zoo.register(
    name="diffusers_clip_model",
    model_fn=partial(transformers.CLIPModel, config=transformers.CLIPConfig()),
    data_gen_fn=data_clip_model,
    output_transform_fn=identity_output,
)

model_zoo.register(
    name="diffusers_clip_text_model",
    model_fn=partial(transformers.CLIPTextModel, config=transformers.CLIPTextConfig()),
    data_gen_fn=data_clip_text,
    output_transform_fn=identity_output,
)

model_zoo.register(
    name="diffusers_clip_vision_model",
    model_fn=partial(transformers.CLIPVisionModel, config=transformers.CLIPVisionConfig()),
    data_gen_fn=data_clip_vision,
    output_transform_fn=clip_vision_model_output,
)

model_zoo.register(
    name="diffusers_unet2d_model",
    model_fn=diffusers.UNet2DModel,
    data_gen_fn=data_unet_fn,
    output_transform_fn=identity_output,
)

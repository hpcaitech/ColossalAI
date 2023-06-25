import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from parse_arguments import parse_args

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext
from colossalai.zero.gemini import get_static_torch_model
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin



if is_wandb_available():
    import wandb

disable_existing_loggers()
logger = get_dist_logger()

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    args = parse_args()

    if args.seed is None:
        colossalai.launch_from_torch(config={})
    else:
        colossalai.launch_from_torch(config={}, seed=args.seed)

    coordinator = DistCoordinator()
    world_size = coordinator.world_size

    booster_kwargs = {}
    if args.plugin == 'torch_ddp_fp16':
        booster_kwargs['mixed_precision'] = 'fp16'
    if args.plugin.startswith('torch_ddp'):
        plugin = TorchDDPPlugin()
    elif args.plugin == 'gemini':
        plugin = GeminiPlugin(placement_policy=args.placement, strict_ddp_mode=True, initial_scale=2 ** 5)
    elif args.plugin == 'low_level_zero':
        plugin = LowLevelZeroPlugin(initial_scale=2 ** 5)

    booster = Booster(plugin=plugin, **booster_kwargs)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    local_rank = coordinator.local_rank
    local_rank = int(local_rank)
    logger.info(f'local_rank: {local_rank}')

    if local_rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if local_rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
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

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # config optimizer for colossalai zero
    optimizer = HybridAdam(unet.parameters(), lr=args.learning_rate, initial_scale=2**5, clipping_norm=args.max_grad_norm)


    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        # Preprocess the captions.
        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions)
        return examples

    with local_rank == 0:
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our booster.
    unet, optimizer, _, _, lr_scheduler = booster.boost(unet, optimizer, lr_scheduler=lr_scheduler)

    if args.use_ema:
        ema_unet.to(get_current_device())

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(get_current_device(), dtype=weight_dtype)
    vae.to(get_current_device(), dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            args.resume_from_checkpoint = None
        else:
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    torch.cuda.synchronize()
    print("start training ... ")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            for key, value in batch.items():
                batch[key] = value.to(get_current_device(), non_blocking=True)

            # We want to learn the denoising process w.r.t the edited images which
            # are conditioned on the original image (which was edited) and the edit instruction.
            # So, first, convert images to latent space.
            latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning.
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Get the additional image embedding for conditioning.
            # Instead of getting a diagonal Gaussian here, we simply take the mode.
            original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

            # Conditioning dropout to support classifier-free guidance during inference. For more details
            # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(bsz, device=latents.device, generator=generator)
                # Sample masks for the edit prompts.
                prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                # Final text conditioning.
                null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                # Sample masks for the original images.
                image_mask_dtype = original_image_embeds.dtype
                image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                )
                image_mask = image_mask.reshape(bsz, 1, 1, 1)
                # Final image conditioning.
                original_image_embeds = image_mask * original_image_embeds

            # Concatenate the `original_image_embeds` with the `noisy_latents`.
            concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1
            progress_bar.update(1)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if args.use_ema:
                ema_unet.step(unet.parameters())
            global_step += 1
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if local_rank == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

            logger.info(f'train_loss : {loss.detach().item()} for global_step : {global_step}')
            logger.info(f'lr: {lr_scheduler.get_last_lr()[0]}')
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    booster.save_model(unet, os.path.join(args.output_dir, "diffusion_pytorch_model.bin"))
    logger.info(f"Saving model checkpoint to {args.output_dir} on rank {local_rank}")

if __name__ == "__main__":
    main()
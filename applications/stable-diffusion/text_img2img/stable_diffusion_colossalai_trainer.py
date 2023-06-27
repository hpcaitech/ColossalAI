import argparse
import logging
import math
import os
import PIL
import random
import shutil
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, PretrainedConfig
from transformers.utils import ContextManagers
from parse_arguments import parse_args
from dreambooth_utils import DreamBoothDataset, PromptDataset, get_full_repo_name

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.loaders import AttnProcsLayers

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


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_tokenizer(config):
    args = config
    if config.task_type == "dreambooth":
        # Load the tokenizer
        if args.tokenizer_name:
            logger.info(f"Loading tokenizer from {args.tokenizer_name}", ranks=[0])
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                revision=args.revision,
                use_fast=False,
            )
        elif args.pretrained_model_name_or_path:
            logger.info("Loading tokenizer from pretrained model", ranks=[0])
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
    
    return tokenizer

def load_text_endcoder(args):
    # import correct text encoder class
    if args.task_type == "dreambooth":
        text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

        # Load models and create wrapper for stable diffusion

        logger.info(f"Loading text_encoder from {args.pretrained_model_name_or_path}", ranks=[0])

        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        

    return text_encoder

def main():
    args = parse_args()

    if args.task_type == "dreambooth":
        assert args.instance_data_dir is not None, "instance_data_dir has to be provided for dreambooth training case"

    DATASET_NAME_MAPPING = {}
    WANDB_TABLE_COL_NAMES = []
    if args.task_type == "image_to_image":
        DATASET_NAME_MAPPING = {
            "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
        }
        WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]
    else:
        DATASET_NAME_MAPPING = {
            "lambdalabs/pokemon-blip-captions": ("image", "text"),
        }


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
    # Make one log on every process with the configuration for debugging.

    if args.seed is not None:
        generator = torch.Generator(device=get_current_device()).manual_seed(args.seed)

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

    
    # Handle the repository creation
    if local_rank == 0:
        if args.output_dir is not None:
            logger.info(f"create output dir : {args.output_dir}")
            os.makedirs(args.output_dir, exist_ok=True)


    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Load the tokenizer
    tokenizer = load_tokenizer(args)
    #load text_encoder 
    text_encoder = load_text_endcoder(args)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    if args.externel_unet_path is None:
        logger.info(f"Loading UNet2DConditionModel from {args.pretrained_model_name_or_path}", ranks=[0])
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,
                                                subfolder="unet",
                                                revision=args.non_ema_revision)
    else:
        logger.info(f"Loading UNet2DConditionModel from {args.externel_unet_path}", ranks=[0])
        unet = UNet2DConditionModel.from_pretrained(args.externel_unet_path,
                                                revision=args.revision,
                                                low_cpu_mem_usage=False)

    if args.task_type == "image_to_image":
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
    if args.use_lora:
        unet.requires_grad_(False)

        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRACrossAttnProcessor(hidden_size=hidden_size,
                                                        cross_attention_dim=cross_attention_dim)

        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)
        

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
    
    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * world_size
        )



    # config optimizer for colossalai zero
    optimizer = HybridAdam(unet.parameters(), lr=args.learning_rate, initial_scale=2**5, clipping_norm=args.max_grad_norm)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.task_type != "dreambooth":
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

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        if args.task_type == "text_to_image":
            if args.image_column is None:
                image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                    )

            if args.caption_column is None:
                caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
            else:
                caption_column = args.caption_column
                if caption_column not in column_names:
                    raise ValueError(
                        f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                    )
        else:
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
    def tokenize_captions_dispatch(task_type):
        def tokenize_captions_1(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids
        
        def tokenize_captions_2(captions):
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids
        
        if task_type == "text_to_image":
            return tokenize_captions_1
        return tokenize_captions_2


    # Preprocessing the datasets.
    if args.task_type == "text_to_image":
        train_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            ]
        )

    tokenize_captions = tokenize_captions_dispatch(args.task_type)
    def process_train_dispatch(task_type):
        def preprocess_train_text_to_image(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples
        
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

        def preprocess_train_image_to_image(examples):
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
        
        if task_type == "text_to_image":
            return preprocess_train_text_to_image
        return preprocess_train_image_to_image
        


    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    if args.task_type == "dreambooth":
        # prepare dataset
        logger.info(f"Prepare dataset from {args.instance_data_dir}", ranks=[0])
        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
        )
    else:

        preprocess_train = process_train_dispatch(args.task_type)
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn_dispatch(task_type):
        def collate_fn_text_to_image(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}
        
        def collate_fn_image_to_image(examples):
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
        
        def collate_fn_dreambooth(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]
    
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = tokenizer.pad(
                {
                    "input_ids": input_ids
                },
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch
            
        if task_type == "text_to_image":
            return collate_fn_text_to_image
        elif task_type == "dreambooth":
            return collate_fn_dreambooth
        else:
            return collate_fn_image_to_image

    # DataLoaders creation:
    collate_fn = collate_fn_dispatch(args.task_type)
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

    if args.use_ema:
        ema_unet.to(get_current_device())


    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(get_current_device(), dtype=weight_dtype)
    vae.to(get_current_device(), dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable = (local_rank!=0))
    progress_bar.set_description("Steps")

    # Use Booster API to use Gemini/Zero with ColossalAI
    unet, optimizer, _, _, lr_scheduler = booster.boost(unet, optimizer, lr_scheduler=lr_scheduler)

    torch.cuda.synchronize()
    print("start training ... ")

    save_flag = False
    for epoch in range(args.num_train_epochs):
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
            
            # Convert images to latent space
            optimizer.zero_grad()
            
            if args.task_type == "text_to_image" or args.task_type == "dreambooth":
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            else:
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )
            if args.input_perturbation:
                new_noise = noise + args.input_perturbation * torch.randn_like(noise)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if args.input_perturbation:
                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
            else:
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            
            if args.task_type == "image_to_image":
                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    if args.seed is not None:
                        random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    else:
                        random_p = torch.rand(bsz, device=latents.device)

                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(get_current_device()))[0]
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
                noisy_latents = concatenated_noisy_latents



            # Get the target for loss depending on the prediction type
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                # rebalance the sample-wise losses with their respective loss weights.
                # Finally, we take the mean of the rebalanced loss.
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()


            
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1
            progress_bar.update(1)
            

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

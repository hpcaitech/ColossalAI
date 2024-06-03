import time

import torch
import transformers
from args import parse_benchmark_args
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam


def format_num(num: int, bytes=False):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def get_data_batch(batch_size, num_labels, num_channels=3, height=224, width=224):
    pixel_values = torch.randn(
        batch_size, num_channels, height, width, device=torch.cuda.current_device(), dtype=torch.float
    )
    labels = torch.randint(0, num_labels, (batch_size,), device=torch.cuda.current_device(), dtype=torch.int64)
    return dict(pixel_values=pixel_values, labels=labels)


def colo_memory_cap(size_in_GB):
    from colossalai.accelerator import get_accelerator
    from colossalai.utils import colo_device_memory_capacity, colo_set_process_memory_fraction

    cuda_capacity = colo_device_memory_capacity(get_accelerator().get_current_device())
    if size_in_GB * (1024**3) < cuda_capacity:
        colo_set_process_memory_fraction(size_in_GB * (1024**3) / cuda_capacity)
        print(f"Limiting GPU memory usage to {size_in_GB} GB")


def main():
    args = parse_benchmark_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(seed=args.seed)
    coordinator = DistCoordinator()
    world_size = coordinator.world_size

    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()
    if coordinator.is_master():
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Whether to set limit on memory capacity
    if args.mem_cap > 0:
        colo_memory_cap(args.mem_cap)

    # Build ViT model
    config = ViTConfig.from_pretrained(args.model_name_or_path)
    model = ViTForImageClassification(config)
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])

    # Enable gradient checkpointing
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # Set plugin
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(offload_optim_frac=1.0, pin_memory=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)
    elif args.plugin == "hybrid_parallel":
        plugin = HybridParallelPlugin(
            tp_size=2,
            pp_size=2,
            num_microbatches=None,
            microbatch_size=1,
            enable_all_optimization=True,
            precision="fp16",
            initial_scale=1,
        )
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(args.learning_rate * world_size))

    # Set criterion (loss function)
    def criterion(outputs, inputs):
        return outputs.loss

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion=criterion)

    # Start training.
    logger.info(f"Start testing", ranks=[0])

    torch.cuda.synchronize()
    model.train()
    start_time = time.time()

    with tqdm(range(args.max_train_steps), desc="Training Step", disable=not coordinator.is_master()) as pbar:
        for _ in pbar:
            optimizer.zero_grad()
            batch = get_data_batch(args.batch_size, args.num_labels, 3, 224, 224)

            if hasattr(booster.plugin, "stage_manager") and booster.plugin.stage_manager is not None:
                # run pipeline forward backward
                batch = iter([batch])
                outputs = booster.execute_pipeline(batch, model, criterion, optimizer, return_loss=True)
            else:
                outputs = model(**batch)
                loss = criterion(outputs, None)
                # Backward
                booster.backward(loss, optimizer)

            optimizer.step()

            torch.cuda.synchronize()

    # Compute Statistics
    end_time = time.time()
    throughput = "{:.4f}".format((world_size * args.max_train_steps * args.batch_size) / (end_time - start_time))
    max_mem = format_num(torch.cuda.max_memory_allocated(device=torch.cuda.current_device()), bytes=True)

    logger.info(
        f"Testing finished, "
        f"batch size per gpu: {args.batch_size}, "
        f"plugin: {args.plugin}, "
        f"throughput: {throughput}, "
        f"maximum memory usage per gpu: {max_mem}.",
        ranks=[0],
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

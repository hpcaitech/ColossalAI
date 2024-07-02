import time
from contextlib import nullcontext

import torch
import tqdm
import transformers
from args import parse_benchmark_args
from transformers import AutoConfig, OPTForCausalLM
from transformers.utils.versions import require_version

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam

require_version("transformers>=4.20.0", "To fix: pip install -r requirements.txt")


def format_num(num: int, bytes=False):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def colo_memory_cap(size_in_GB):
    from colossalai.utils import colo_device_memory_capacity, colo_set_process_memory_fraction, get_current_device

    cuda_capacity = colo_device_memory_capacity(get_current_device())
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

    # Whether to set limit of memory capacity
    if args.mem_cap > 0:
        colo_memory_cap(args.mem_cap)

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
    logger.info(f"Set plugin as {args.plugin}", ranks=[0])

    # Build OPT model
    init_ctx = (
        LazyInitContext(default_device=get_accelerator().get_current_device())
        if isinstance(plugin, (GeminiPlugin))
        else nullcontext()
    )
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    with init_ctx:
        model = OPTForCausalLM(config=config)
    logger.info(f"Finish loading model from {args.model_name_or_path}", ranks=[0])

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=args.learning_rate)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    # Start training.
    logger.info(f"Start testing", ranks=[0])
    progress_bar = tqdm.tqdm(total=args.max_train_steps, desc="Training Step", disable=not coordinator.is_master())

    torch.cuda.synchronize()
    model.train()
    start_time = time.time()

    for _ in range(args.max_train_steps):
        input_ids, attn_mask = get_data(args.batch_size, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, use_cache=False)
        loss = outputs["loss"]
        booster.backward(loss, optimizer)
        optimizer.step()

        torch.cuda.synchronize()
        progress_bar.update(1)

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


if __name__ == "__main__":
    main()

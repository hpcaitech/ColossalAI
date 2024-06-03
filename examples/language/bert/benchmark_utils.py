import inspect
from logging import getLogger
from time import time
from typing import Callable

import torch
import yaml
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator

logger = getLogger("colossalai-booster-benchmark")
_INVALID = float("nan")


def format_num(num: int, bytes=False):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def _is_valid(val):
    return val == val


def get_call_arg_names(module_or_fn):
    if isinstance(module_or_fn, torch.nn.Module):
        return inspect.getfullargspec(module_or_fn.forward)[0][1:]
    return inspect.getfullargspec(module_or_fn)[0]


def measure_params(model):
    num_params = _INVALID

    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError as e:
        logger.error(f"Unable to measure model params due to error: {e}")

    return num_params


def warm_up(
    model,
    booster,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler,
    num_runs=10,
):
    for i, data in enumerate(dataloader):
        if i > num_runs:
            break
        inputs, labels = data[0].cuda(), data[1].cuda()
        outputs = model(inputs, labels=labels)
        loss = criterion(outputs)
        booster.backward(loss, optimizer)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


def fmt(d: dict):
    return yaml.dump(d)


def benchmark(
    model: torch.nn.Module,
    booster: Booster,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler,
    dataloader: DataLoader,
    criterion: Callable = None,
    warm_up_fn=warm_up,
    epoch_num: int = 3,
    batch_size: int = 32,
    warm_up_steps: int = 3,
):
    results = {}
    model_device = torch.cuda.current_device()

    # Warm up
    warm_up_fn(
        model,
        booster,
        dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        num_runs=warm_up_steps,
    )
    # Measure params
    params = measure_params(model)
    if _is_valid(params):
        results["params"] = format_num(params)
        logger.info(f"Model parameters: {params} ({format_num(params)})")

    # Measure Allocated Memory and Throughput
    memory = {}
    throughput = {}
    torch.cuda.reset_peak_memory_stats(device=model_device)
    pre_mem = torch.cuda.memory_allocated(device=model_device)

    start_time = time()

    for epoch in range(epoch_num):
        with tqdm(
            dataloader, desc=f"Epoch [{epoch + 1}/{epoch_num}]", disable=not DistCoordinator().is_master()
        ) as pbar:
            for data in pbar:
                inputs, labels = data[0].cuda(), data[1].cuda()
                outputs = model(inputs, labels=labels)
                loss = criterion(outputs)
                booster.backward(loss, optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    end_time = time()

    all_sample = epoch_num * len(dataloader)

    post_mem = torch.cuda.memory_allocated(device=model_device)
    max_mem = torch.cuda.max_memory_allocated(device=model_device)

    memory[f"batch_size_{batch_size}"] = {
        "cuda_pre_training_bytes": format_num(pre_mem, bytes=True),
        "cuda_max_training_bytes": format_num(max_mem, bytes=True),
        "cuda_post_training_bytes": format_num(post_mem, bytes=True),
    }
    logger.info(fmt({f"Memory results (batch_size={batch_size})": memory[f"batch_size_{batch_size}"]}))

    throughput[f"batch_size_{batch_size}"] = {
        "throughput:": "{:.1f}".format(all_sample * DistCoordinator().world_size / (end_time - start_time))
    }
    logger.info(fmt({f"Throughput results (batch_size={batch_size})": throughput[f"batch_size_{batch_size}"]}))

    results["throughput"] = throughput
    results["memory"] = memory

    return results

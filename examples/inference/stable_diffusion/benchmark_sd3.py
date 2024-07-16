import argparse
import time
from contextlib import nullcontext
from typing import List, Union

import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

import colossalai
from colossalai.inference.config import DiffusionGenerationConfig, InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

GIGABYTE = 1024**3
MEGABYTE = 1024 * 1024

_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def log_generation_time(log_message, log_file):
    with open(log_file, "a") as f:
        f.write(log_message)


def benchmark_colossalai(rank, world_size, port, args):
    if isinstance(args.width, int):
        width_list = [args.width]
    else:
        width_list = args.width

    if isinstance(args.height, int):
        height_list = [args.height]
    else:
        height_list = args.height

    assert len(width_list) == len(height_list)

    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    from colossalai.cluster.dist_coordinator import DistCoordinator

    coordinator = DistCoordinator()

    inference_config = InferenceConfig(
        dtype=args.dtype,
        patched_parallelism_size=args.patched_parallel_size,
    )
    engine = InferenceEngine(args.model, inference_config=inference_config, verbose=False)

    # warmup
    for i in range(args.n_warm_up_steps):
        engine.generate(
            prompts=["hello world"],
            generation_config=DiffusionGenerationConfig(
                num_inference_steps=args.num_inference_steps, height=1024, width=1024
            ),
        )

    ctx = (
        torch.profiler.profile(
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        if args.profile
        else nullcontext()
    )

    for h, w in zip(height_list, width_list):
        with ctx as prof:
            start = time.perf_counter()
            for i in range(args.n_repeat_times):
                engine.generate(
                    prompts=["hello world"],
                    generation_config=DiffusionGenerationConfig(
                        num_inference_steps=args.num_inference_steps, height=h, width=w
                    ),
                )
            end = time.perf_counter()
        log_msg = f"[ColossalAI]avg generation time for h({h})xw({w}) is {(end - start) / args.n_repeat_times:.2f}s"
        coordinator.print_on_master(log_msg)
        if args.log:
            log_file = f"examples/inference/stable_diffusion/benchmark_bs{args.batch_size}_pps{args.patched_parallel_size}_steps{args.num_inference_steps}_height{h}_width{w}_dtype{args.dtype}_profile{args.profile}_model{args.model.split('/')[-1]}_mode{args.mode}.log"
            if dist.get_rank() == 0:
                log_generation_time(log_message=log_msg, log_file=log_file)

        if args.profile:
            file = f"examples/inference/stable_diffusion/benchmark_bs{args.batch_size}_pps{args.patched_parallel_size}_steps{args.num_inference_steps}_height{h}_width{w}_dtype{args.dtype}_warmup{args.n_warm_up_steps}_repeat{args.n_repeat_times}_profile{args.profile}_model{args.model.split('/')[-1]}_mode{args.mode}_rank_{dist.get_rank()}.json"
            prof.export_chrome_trace(file)


def benchmark_diffusers(args):
    if isinstance(args.width, int):
        width_list = [args.width]
    else:
        width_list = args.width

    if isinstance(args.height, int):
        height_list = [args.height]
    else:
        height_list = args.height

    assert len(width_list) == len(height_list)

    model = DiffusionPipeline.from_pretrained(args.model, torch_dtype=_DTYPE_MAPPING[args.dtype]).to("cuda")

    # warmup
    for i in range(args.n_warm_up_steps):
        model(prompt="hello world", num_inference_steps=args.num_inference_steps, height=1024, width=1024)

    ctx = (
        torch.profiler.profile(
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        if args.profile
        else nullcontext()
    )

    for h, w in zip(height_list, width_list):
        with ctx as prof:
            start = time.perf_counter()
            for i in range(args.n_repeat_times):
                model(prompt="hello world", num_inference_steps=args.num_inference_steps, height=h, width=w)
            end = time.perf_counter()
        log_msg = f"[Diffusers]avg generation time for h({h})xw({w}) is {(end - start) / args.n_repeat_times:.2f}s"
        print(log_msg)
        if args.log:
            log_file = f"examples/inference/stable_diffusion/benchmark_bs{args.batch_size}_pps{args.patched_parallel_size}_steps{args.num_inference_steps}_height{h}_width{w}_dtype{args.dtype}_profile{args.profile}_model{args.model.split('/')[-1]}_mode{args.mode}.log"
            log_generation_time(log_message=log_msg, log_file=log_file)

        if args.profile:
            file = f"examples/inference/stable_diffusion/benchmark_bs{args.batch_size}_pps{args.patched_parallel_size}_steps{args.num_inference_steps}_height{h}_width{w}_dtype{args.dtype}_warmup{args.n_warm_up_steps}_repeat{args.n_repeat_times}_profile{args.profile}_model{args.model.split('/')[-1]}_mode{args.mode}.json"
            prof.export_chrome_trace(file)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def benchmark(args):
    if args.mode == "colossalai":
        spawn(benchmark_colossalai, nprocs=args.patched_parallel_size, args=args)
    elif args.mode == "diffusers":
        benchmark_diffusers(args)


# CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 2 && python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" -p 2 --mode colossalai --log
# CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 4 && python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" -p 4 --mode colossalai --log
# CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 8 && python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" -p 8 --mode colossalai --log
# CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 1 && python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" --mode diffusers --log

# enable profiler
# CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 2 && python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" -p 2 --mode colossalai --n_warm_up_steps 3 --n_repeat_times 1 --profile --num_inference_steps 20
# CUDA_VISIBLE_DEVICES_set_n_least_memory_usage 1 && python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" --mode diffusers --n_warm_up_steps 3 --n_repeat_times 1 --profile --num_inference_steps 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-p", "--patched_parallel_size", type=int, default=1, help="Patched Parallelism size")
    parser.add_argument("-n", "--num_inference_steps", type=int, default=50, help="num of inference steps")
    parser.add_argument("-H", "--height", type=Union[int, List[int]], default=[1024, 2048, 3840], help="Height List")
    parser.add_argument("-w", "--width", type=Union[int, List[int]], default=[1024, 2048, 3840], help="Width List")
    parser.add_argument("--dtype", type=str, default="fp16", help="data type", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--n_warm_up_steps", type=int, default=3, help="warm up times")
    parser.add_argument("--n_repeat_times", type=int, default=5, help="repeat times")
    parser.add_argument("--profile", default=False, action="store_true", help="enable torch profiler")
    parser.add_argument("--log", default=False, action="store_true", help="enable torch profiler")
    parser.add_argument(
        "-m",
        "--model",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="the type of model",
        # choices=["stabilityai/stable-diffusion-3-medium-diffusers", "PixArt-alpha/PixArt-XL-2-1024-MS"],
    )
    parser.add_argument(
        "--mode",
        default="colossalai",
        # choices=["colossalai", "diffusers"],
        help="decide which inference framework to run",
    )
    args = parser.parse_args()
    benchmark(args)

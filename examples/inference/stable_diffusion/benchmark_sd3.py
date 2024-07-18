import argparse
import json
import time
from contextlib import nullcontext

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


def log_generation_time(log_data, log_file):
    with open(log_file, "a") as f:
        json.dump(log_data, f, indent=2)
        f.write("\n")


def warmup(engine, args):
    for _ in range(args.n_warm_up_steps):
        engine.generate(
            prompts=["hello world"],
            generation_config=DiffusionGenerationConfig(
                num_inference_steps=args.num_inference_steps, height=args.height[0], width=args.width[0]
            ),
        )


def profile_context(args):
    return (
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


def log_and_profile(h, w, avg_time, log_msg, args, model_name, mode, prof=None):
    log_data = {
        "mode": mode,
        "model": model_name,
        "batch_size": args.batch_size,
        "patched_parallel_size": args.patched_parallel_size,
        "num_inference_steps": args.num_inference_steps,
        "height": h,
        "width": w,
        "dtype": args.dtype,
        "profile": args.profile,
        "n_warm_up_steps": args.n_warm_up_steps,
        "n_repeat_times": args.n_repeat_times,
        "avg_generation_time": avg_time,
        "log_message": log_msg,
    }

    if args.log:
        log_file = f"examples/inference/stable_diffusion/benchmark_{model_name}_{mode}.json"
        log_generation_time(log_data=log_data, log_file=log_file)

    if args.profile:
        file = f"examples/inference/stable_diffusion/benchmark_{model_name}_{mode}_prof.json"
        prof.export_chrome_trace(file)


def benchmark_colossalai(rank, world_size, port, args):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    from colossalai.cluster.dist_coordinator import DistCoordinator

    coordinator = DistCoordinator()

    inference_config = InferenceConfig(
        dtype=args.dtype,
        patched_parallelism_size=args.patched_parallel_size,
    )
    engine = InferenceEngine(args.model, inference_config=inference_config, verbose=False)

    warmup(engine, args)

    for h, w in zip(args.height, args.width):
        with profile_context(args) as prof:
            start = time.perf_counter()
            for _ in range(args.n_repeat_times):
                engine.generate(
                    prompts=["hello world"],
                    generation_config=DiffusionGenerationConfig(
                        num_inference_steps=args.num_inference_steps, height=h, width=w
                    ),
                )
            end = time.perf_counter()

        avg_time = (end - start) / args.n_repeat_times
        log_msg = f"[ColossalAI]avg generation time for h({h})xw({w}) is {avg_time:.2f}s"
        coordinator.print_on_master(log_msg)

        if dist.get_rank() == 0:
            log_and_profile(h, w, avg_time, log_msg, args, args.model.split("/")[-1], "colossalai", prof=prof)


def benchmark_diffusers(args):
    model = DiffusionPipeline.from_pretrained(args.model, torch_dtype=_DTYPE_MAPPING[args.dtype]).to("cuda")

    for _ in range(args.n_warm_up_steps):
        model(
            prompt="hello world",
            num_inference_steps=args.num_inference_steps,
            height=args.height[0],
            width=args.width[0],
        )

    for h, w in zip(args.height, args.width):
        with profile_context(args) as prof:
            start = time.perf_counter()
            for _ in range(args.n_repeat_times):
                model(prompt="hello world", num_inference_steps=args.num_inference_steps, height=h, width=w)
            end = time.perf_counter()

        avg_time = (end - start) / args.n_repeat_times
        log_msg = f"[Diffusers]avg generation time for h({h})xw({w}) is {avg_time:.2f}s"
        print(log_msg)

        log_and_profile(h, w, avg_time, log_msg, args, args.model.split("/")[-1], "diffusers", prof)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def benchmark(args):
    if args.mode == "colossalai":
        spawn(benchmark_colossalai, nprocs=args.patched_parallel_size, args=args)
    elif args.mode == "diffusers":
        benchmark_diffusers(args)


"""
# enable log
python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" -p 2 --mode colossalai --log
python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" --mode diffusers --log

# enable profiler
python examples/inference/stable_diffusion/benchmark_sd3.py -m "stabilityai/stable-diffusion-3-medium-diffusers" -p 2 --mode colossalai --n_warm_up_steps 3 --n_repeat_times 1 --profile --num_inference_steps 20
python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" -p 2 --mode colossalai --n_warm_up_steps 3 --n_repeat_times 1 --profile --num_inference_steps 20
python examples/inference/stable_diffusion/benchmark_sd3.py -m "PixArt-alpha/PixArt-XL-2-1024-MS" --mode diffusers --n_warm_up_steps 3 --n_repeat_times 1 --profile --num_inference_steps 20
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-p", "--patched_parallel_size", type=int, default=1, help="Patched Parallelism size")
    parser.add_argument("-n", "--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("-H", "--height", type=int, nargs="+", default=[1024, 2048], help="Height list")
    parser.add_argument("-w", "--width", type=int, nargs="+", default=[1024, 2048], help="Width list")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Data type")
    parser.add_argument("--n_warm_up_steps", type=int, default=3, help="Number of warm up steps")
    parser.add_argument("--n_repeat_times", type=int, default=5, help="Number of repeat times")
    parser.add_argument("--profile", default=False, action="store_true", help="Enable torch profiler")
    parser.add_argument("--log", default=False, action="store_true", help="Enable logging")
    parser.add_argument("-m", "--model", default="stabilityai/stable-diffusion-3-medium-diffusers", help="Model path")
    parser.add_argument(
        "--mode", default="colossalai", choices=["colossalai", "diffusers"], help="Inference framework mode"
    )
    args = parser.parse_args()
    benchmark(args)

from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


@torch.no_grad()
def all_reduce_mean(x: float, world_size: int) -> float:
    if world_size == 1:
        return x
    # BUG: RuntimeError: Invalid scalar type when use dist.all_reduce(tensor, group=gloo_group)
    # # Use CPU tensor to avoid OOM/weird NCCl error
    # gloo_group = dist.new_group(backend="gloo")
    # tensor = torch.tensor([x], device="cpu")
    # dist.all_reduce(tensor, group=gloo_group)
    # tensor = tensor / world_size
    # return tensor.item()

    tensor = torch.tensor([x], device=get_current_device(), dtype=torch.float)
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


def get_profile_context(enable_flag, warmup_steps, active_steps, save_dir, nsys=False):
    class DummyProfiler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    class NsysProfiler:
        def __init__(self, warmup_steps, active_steps):
            self.step_number = 0
            self.warmup_steps = warmup_steps
            self.active_steps = active_steps

        def step(self):
            if self.step_number == self.warmup_steps:
                torch.cuda.cudart().cudaProfilerStart()
            elif self.step_number == self.warmup_steps + self.active_steps:
                torch.cuda.cudart().cudaProfilerStop()
            self.step_number += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    if enable_flag:
        if nsys:
            return NsysProfiler(warmup_steps, active_steps)

        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps),
            on_trace_ready=tensorboard_trace_handler(save_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        return DummyProfiler()


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration += time() - self.start_time
        self.start_time = None

    def reset(self) -> None:
        self.duration = 0.0


class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        model_numel: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        enable_grad_checkpoint: bool = False,
        ignore_steps: int = 0,
        dp_world_size: Optional[int] = None,
    ) -> None:
        self.model_numel = model_numel
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_steps = ignore_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.coordinator = DistCoordinator()
        self.dp_world_size = dp_world_size or self.coordinator.world_size
        self.disable: bool = False
        self.timer = Timer()
        self.num_samples: int = 0
        self.flop_megatron = 0
        self.flop: int = 0

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        if self.disable:
            return
        # get_accelerator().synchronize()
        self.timer.start()

    def on_step_end(self, input_ids: Tensor, **kwargs) -> None:
        if self.disable:
            return
        # get_accelerator().synchronize()
        self.timer.end()

        batch_size, seq_len = input_ids.shape

        self.num_samples += batch_size
        checkpoint_activations_factor = 3 + int(self.enable_grad_checkpoint)
        self.flop_megatron += (
            24 * checkpoint_activations_factor * batch_size * seq_len * self.num_layers * (self.hidden_size**2)
        ) * (
            1.0 + (seq_len / (6.0 * self.hidden_size)) + (self.vocab_size / (16.0 * self.num_layers * self.hidden_size))
        )
        self.flop += batch_size * seq_len * self.model_numel * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_fit_end(self) -> None:
        avg_duration = all_reduce_mean(self.timer.duration, self.coordinator.world_size)
        avg_throughput = self.num_samples * self.dp_world_size / (avg_duration + 1e-12)
        mp_world_size = self.coordinator.world_size // self.dp_world_size
        avg_tflops_per_gpu_megatron = self.flop_megatron / 1e12 / (avg_duration + 1e-12) / mp_world_size
        avg_tflops_per_gpu = self.flop / 1e12 / (avg_duration + 1e-12) / mp_world_size
        self.coordinator.print_on_master(
            f"num_samples: {self.num_samples}, dp_world_size: {self.dp_world_size}, flop_megatron: {self.flop_megatron}, flop: {self.flop}, avg_duration: {avg_duration}, "
            f"avg_throughput: {avg_throughput}"
        )
        self.coordinator.print_on_master(
            f"Throughput: {avg_throughput:.2f} samples/sec, TFLOPS per GPU by Megatron: {avg_tflops_per_gpu_megatron:.2f}, TFLOPS per GPU: {avg_tflops_per_gpu:.2f}"
        )

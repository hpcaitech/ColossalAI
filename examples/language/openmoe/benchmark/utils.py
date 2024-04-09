from time import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from colossalai.logging import DistributedLogger


def print_model_numel(logger: DistributedLogger, model: nn.Module) -> None:
    B = 1024**3
    M = 1024**2
    K = 1024
    outputs = "Model param count: "
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_param >= B:
        outputs += f"{model_param / B:.2f} B\n"
    elif model_param >= M:
        outputs += f"{model_param / M:.2f} M\n"
    elif model_param >= K:
        outputs += f"{model_param / K:.2f} K\n"
    else:
        outputs += f"{model_param}\n"
    logger.info(outputs, ranks=[0])


def get_model_numel(model: nn.Module) -> None:
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_param


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
    tensor = torch.tensor([x], device=torch.cuda.current_device())
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


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
        enable_grad_checkpoint: bool = False,
        ignore_steps: int = 0,
        dp_world_size: Optional[int] = None,
    ) -> None:
        self.model_numel = model_numel
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_steps = ignore_steps
        self.dp_world_size = dp_world_size
        self.world_size = dist.get_world_size()
        self.disable: bool = False
        self.timer = Timer()
        self.num_samples: int = 0
        self.flop: int = 0

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        if self.disable:
            return
        torch.cuda.synchronize()
        self.timer.start()

    def on_step_end(self, input_ids: Tensor, **kwargs) -> None:
        if self.disable:
            return
        torch.cuda.synchronize()
        self.timer.end()

        batch_size, seq_len = input_ids.shape

        self.num_samples += batch_size
        self.flop += batch_size * seq_len * self.model_numel * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_fit_end(self) -> None:
        avg_duration = all_reduce_mean(self.timer.duration, self.world_size)
        avg_throughput = self.num_samples * self.dp_world_size / (avg_duration + 1e-12)
        mp_world_size = self.world_size // self.dp_world_size
        avg_tflops_per_gpu = self.flop / 1e12 / (avg_duration + 1e-12) / mp_world_size
        if dist.get_rank() == 0:
            print(
                f"num_samples: {self.num_samples}, dp_world_size: {self.dp_world_size}, flop: {self.flop}, avg_duration: {avg_duration}, "
                f"avg_throughput: {avg_throughput}"
            )
            print(f"Throughput: {avg_throughput:.2f} samples/sec, TFLOPS per GPU: {avg_tflops_per_gpu:.2f}")

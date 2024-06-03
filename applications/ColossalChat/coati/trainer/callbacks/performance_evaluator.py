from time import time
from typing import Optional

import torch
import torch.distributed as dist
from coati.experience_maker import Experience

from .base import Callback


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def save_eval_result_rank_0(s: str, save_path: str, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(save_path, "a+") as f:
            train_config = "; ".join([str(kwargs[key]) for key in kwargs])
            f.write(train_config + "\n" + s + "\n")


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


class PerformanceEvaluator(Callback):
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
        actor_num_params: int,
        critic_num_params: int,
        initial_model_num_params: int,
        reward_model_num_params: int,
        enable_grad_checkpoint: bool = False,
        ignore_episodes: int = 0,
        train_config: Optional[dict] = None,
        save_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.world_size = get_world_size()
        self.actor_num_params = actor_num_params
        self.critic_num_params = critic_num_params
        self.initial_model_num_params = initial_model_num_params
        self.reward_model_num_params = reward_model_num_params
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_episodes = ignore_episodes
        self.disable: bool = False

        self.overall_timer = Timer()
        self.make_experience_timer = Timer()
        self.learn_timer = Timer()
        self.make_experience_num_samples: int = 0
        self.make_experience_flop: int = 0
        self.learn_num_samples: int = 0
        self.learn_flop: int = 0
        self.train_config = train_config
        self.save_path = save_path

    def on_episode_start(self, episode: int) -> None:
        self.disable = self.ignore_episodes > 0 and episode < self.ignore_episodes
        if self.disable:
            return
        self.overall_timer.start()

    def on_episode_end(self, episode: int) -> None:
        if self.disable:
            return
        self.overall_timer.end()

    def on_make_experience_start(self) -> None:
        if self.disable:
            return
        self.make_experience_timer.start()

    def on_make_experience_end(self, experience: Experience) -> None:
        if self.disable:
            return
        self.make_experience_timer.end()

        batch_size, seq_len = experience.sequences.shape

        self.make_experience_num_samples += batch_size

        # actor generate
        num_actions = experience.action_mask.size(1)
        input_len = seq_len - num_actions
        total_seq_len = (input_len + seq_len - 1) * num_actions / 2
        self.make_experience_flop += self.actor_num_params * batch_size * total_seq_len * 2
        # actor forward
        self.make_experience_flop += self.actor_num_params * batch_size * seq_len * 2
        # critic forward
        self.make_experience_flop += self.critic_num_params * batch_size * seq_len * 2
        # initial model forward
        self.make_experience_flop += self.initial_model_num_params * batch_size * seq_len * 2
        # reward model forward
        self.make_experience_flop += self.reward_model_num_params * batch_size * seq_len * 2

    def on_learn_batch_start(self) -> None:
        if self.disable:
            return
        self.learn_timer.start()

    def on_learn_batch_end(self, experience: Experience) -> None:
        if self.disable:
            return
        self.learn_timer.end()

        batch_size, seq_len = experience.sequences.shape

        self.learn_num_samples += batch_size

        # actor forward-backward, 3 means forward(1) + backward(2)
        self.learn_flop += self.actor_num_params * batch_size * seq_len * 2 * (3 + int(self.enable_grad_checkpoint))
        # critic forward-backward
        self.learn_flop += self.critic_num_params * batch_size * seq_len * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_fit_end(self) -> None:
        avg_make_experience_duration = all_reduce_mean(self.make_experience_timer.duration, self.world_size)
        avg_learn_duration = all_reduce_mean(self.learn_timer.duration, self.world_size)
        avg_overall_duration = all_reduce_mean(self.overall_timer.duration, self.world_size)

        avg_make_experience_throughput = (
            self.make_experience_num_samples * self.world_size / (avg_make_experience_duration + 1e-12)
        )
        avg_make_experience_tflops = self.make_experience_flop / 1e12 / (avg_make_experience_duration + 1e-12)

        avg_learn_throughput = self.learn_num_samples * self.world_size / (avg_learn_duration + 1e-12)
        avg_learn_tflops = self.learn_flop / 1e12 / (avg_learn_duration + 1e-12)

        num_effective_samples = min(self.learn_num_samples, self.make_experience_num_samples) * self.world_size

        avg_overall_throughput = num_effective_samples / (avg_overall_duration + 1e-12)

        overall_time_per_sample = divide(1, avg_overall_throughput)
        make_experience_time_per_sample = divide(avg_make_experience_duration, num_effective_samples)
        learn_time_per_sample = divide(avg_learn_duration, num_effective_samples)

        save_eval_result_rank_0(
            f"Performance summary:\n"
            + f"Generate {self.make_experience_num_samples * self.world_size} samples, throughput: {avg_make_experience_throughput:.2f} samples/s, TFLOPS per GPU: {avg_make_experience_tflops:.2f}\n"
            + f"Train {self.learn_num_samples * self.world_size} samples, throughput: {avg_learn_throughput:.2f} samples/s, TFLOPS per GPU: {avg_learn_tflops:.2f}\n"
            + f"Overall throughput: {avg_overall_throughput:.2f} samples/s\n"
            + f"Overall time per sample: {overall_time_per_sample:.2f} s\n"
            + f"Make experience time per sample: {make_experience_time_per_sample:.2f} s, {make_experience_time_per_sample/overall_time_per_sample*100:.2f}%\n"
            + f"Learn time per sample: {learn_time_per_sample:.2f} s, {learn_time_per_sample/overall_time_per_sample*100:.2f}%",
            self.save_path,
            **self.train_config,
        )

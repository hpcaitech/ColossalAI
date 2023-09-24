from time import time
from typing import Optional

import torch
import torch.distributed as dist
from coati.experience_maker import Experience

from .base import MakerCallback, TrainerCallback


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def print_rank_0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


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
        self.duration += time() - self.start_time

    def reset(self) -> None:
        self.duration = 0.0


class ExperienceMakerPerformanceEvaluator(MakerCallback):
    def __init__(
        self, actor_num_params: int, critic_num_params: int, initial_model_num_params: int, reward_model_num_params: int
    ) -> None:
        super().__init__()
        self.world_size = get_world_size()
        self.actor_num_params = actor_num_params
        self.critic_num_params = critic_num_params
        self.initial_model_num_params = initial_model_num_params
        self.reward_model_num_params = reward_model_num_params

        self.batch_timer = Timer()
        self.send_timer = Timer()
        self.make_experience_timer = Timer()
        self.total_samples: int = 0
        self.make_experience_flop: int = 0

        print_rank_0(
            f"ExperienceMaker actor: {actor_num_params/1024**3:.2f}B, critic: {critic_num_params/1024**3:.2f}B, initial model: {initial_model_num_params/1024**3:.2f}B, reward model: {reward_model_num_params/1024**3:.2f}B, world size: {self.world_size}"
        )

    def on_make_experience_start(self) -> None:
        self.make_experience_timer.start()

    def on_make_experience_end(self, experience: Experience) -> None:
        self.make_experience_timer.end()

        batch_size, seq_len = experience.sequences.shape

        self.total_samples += batch_size

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

    def on_send_start(self) -> None:
        self.send_timer.start()

    def on_send_end(self) -> None:
        self.send_timer.end()

    def on_batch_start(self) -> None:
        self.batch_timer.start()

    def on_batch_end(self) -> None:
        self.batch_timer.end()

    def on_loop_end(self) -> None:
        avg_make_experience_duration = all_reduce_mean(self.make_experience_timer.duration, self.world_size)
        avg_overall_duration = all_reduce_mean(self.batch_timer.duration, self.world_size)
        avg_send_duration = all_reduce_mean(self.send_timer.duration, self.world_size)

        avg_throughput = self.total_samples * self.world_size / (avg_overall_duration + 1e-12)
        avg_make_experience_tflops = self.make_experience_flop / 1e12 / (avg_make_experience_duration + 1e-12)
        avg_time_per_sample = (avg_overall_duration + 1e-12) / (self.total_samples * self.world_size)
        avg_make_experience_time_per_sample = (avg_make_experience_duration + 1e-12) / (
            self.total_samples * self.world_size
        )
        avg_send_time_per_sample = (avg_send_duration + 1e-12) / (self.total_samples * self.world_size)

        print_rank_0(
            "Making Experience Performance Summary:\n"
            + f"Throughput: {avg_throughput:.3f} samples/sec\n"
            + f"TFLOPS per GPU: {avg_make_experience_tflops:.3f}\n"
            + f"Sample time (overall): {avg_time_per_sample:.3f} s\n"
            + f"Sample time (make experience): {avg_make_experience_time_per_sample:.3f} s, {avg_make_experience_time_per_sample/avg_time_per_sample*100:.2f}%\n"
            + f"Sample time (send): {avg_send_time_per_sample:.3f} s, {avg_send_time_per_sample/avg_time_per_sample*100:.2f}%\n"
        )


class TrainerPerformanceEvaluator(TrainerCallback):
    def __init__(
        self,
        actor_num_params: int,
        critic_num_params: int,
        enable_grad_checkpoint: bool = False,
        ignore_first_episodes: int = 1,
    ) -> None:
        super().__init__()
        self.world_size = get_world_size()
        self.actor_num_params = actor_num_params
        self.critic_num_params = critic_num_params
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_first_episodes = ignore_first_episodes
        self.ignore_this_episode = False

        self.episode_timer = Timer()
        self.batch_timer = Timer()
        self.update_timer = Timer()
        self.total_samples: int = 0
        self.learn_flop: int = 0

        print_rank_0(
            f"Trainer actor: {self.actor_num_params/1024**3:.2f}B, critic: {self.critic_num_params/1024**3:.2f}B, world size: {self.world_size}"
        )

    def on_episode_start(self, episodes: int) -> None:
        self.ignore_this_episode = episodes < self.ignore_first_episodes
        if self.ignore_this_episode:
            return
        self.episode_timer.start()

    def on_episode_end(self, episodes: int) -> None:
        if self.ignore_this_episode:
            return
        self.episode_timer.end()

    def on_batch_start(self) -> None:
        if self.ignore_this_episode:
            return
        self.batch_timer.start()

    def on_batch_end(self, metrics: dict, experience: Experience) -> None:
        if self.ignore_this_episode:
            return
        self.batch_timer.end()

        batch_size, seq_len = experience.sequences.shape

        self.total_samples += batch_size

        # actor forward-backward, 3 means forward(1) + backward(2)
        self.learn_flop += self.actor_num_params * batch_size * seq_len * 2 * (3 + int(self.enable_grad_checkpoint))
        # critic forward-backward
        self.learn_flop += self.critic_num_params * batch_size * seq_len * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_update_start(self) -> None:
        if self.ignore_this_episode:
            return
        self.update_timer.start()

    def on_update_end(self) -> None:
        if self.ignore_this_episode:
            return
        self.update_timer.end()

    def on_fit_end(self) -> None:
        if self.total_samples == 0:
            print_rank_0("No samples are collected, skip trainer performance evaluation")
            return
        avg_train_duration = all_reduce_mean(self.batch_timer.duration, self.world_size)
        avg_update_duration = all_reduce_mean(self.update_timer.duration, self.world_size)
        avg_episode_duration = all_reduce_mean(self.episode_timer.duration, self.world_size)

        avg_throughput = self.total_samples * self.world_size / (avg_episode_duration + 1e-12)
        avg_learn_tflops = self.learn_flop / 1e12 / (avg_train_duration + 1e-12)
        avg_time_per_sample = (avg_episode_duration + 1e-12) / (self.total_samples * self.world_size)
        avg_train_time_per_sample = (avg_train_duration + 1e-12) / (self.total_samples * self.world_size)
        avg_update_time_per_sample = (avg_update_duration + 1e-12) / (self.total_samples * self.world_size)

        print_rank_0(
            "Learning Performance Summary:\n"
            + f"Throughput: {avg_throughput:.3f} samples/sec\n"
            + f"TFLOPS per GPU: {avg_learn_tflops:.3f}\n"
            + f"Sample time (overall): {avg_time_per_sample:.3f} s\n"
            + f"Sample time (train): {avg_train_time_per_sample:.3f} s, {avg_train_time_per_sample/avg_time_per_sample*100:.2f}%\n"
            + f"Sample time (update): {avg_update_time_per_sample:.3f} s, {avg_update_time_per_sample/avg_time_per_sample*100:.2f}%\n"
        )

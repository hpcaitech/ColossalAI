from time import time
from typing import Optional

import torch
import torch.distributed as dist
from chatgpt.experience_maker import Experience

from .base import Callback


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

    def __init__(self,
                 actor_num_params: int,
                 critic_num_params: int,
                 initial_model_num_params: int,
                 reward_model_num_params: int,
                 enable_grad_checkpoint: bool = False,
                 ignore_episodes: int = 0) -> None:
        super().__init__()
        self.world_size = get_world_size()
        self.actor_num_params = actor_num_params
        self.critic_num_params = critic_num_params
        self.initial_model_num_params = initial_model_num_params
        self.reward_model_num_params = reward_model_num_params
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_episodes = ignore_episodes
        self.disable: bool = False

        self.make_experience_duration: float = 0.
        self.make_experience_start_time: Optional[float] = None
        self.make_experience_num_samples: int = 0
        self.make_experience_flop: int = 0
        self.learn_duration: float = 0.
        self.learn_start_time: Optional[float] = None
        self.learn_num_samples: int = 0
        self.learn_flop: int = 0

    def on_episode_start(self, episode: int) -> None:
        self.disable = self.ignore_episodes > 0 and episode < self.ignore_episodes

    def on_make_experience_start(self) -> None:
        if self.disable:
            return
        self.make_experience_start_time = time()

    def on_make_experience_end(self, experience: Experience) -> None:
        if self.disable:
            return
        self.make_experience_duration += time() - self.make_experience_start_time

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
        self.learn_start_time = time()

    def on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        if self.disable:
            return
        self.learn_duration += time() - self.learn_start_time

        batch_size, seq_len = experience.sequences.shape

        self.learn_num_samples += batch_size

        # actor forward-backward, 3 means forward(1) + backward(2)
        self.learn_flop += self.actor_num_params * batch_size * seq_len * 2 * (3 + int(self.enable_grad_checkpoint))
        # critic foward-backward
        self.learn_flop += self.critic_num_params * batch_size * seq_len * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_fit_end(self) -> None:
        avg_make_experience_duration = all_reduce_mean(self.make_experience_duration, self.world_size)
        avg_learn_duration = all_reduce_mean(self.learn_duration, self.world_size)

        avg_make_experience_throughput = self.make_experience_num_samples / (avg_make_experience_duration + 1e-12)
        avg_make_experience_tflops = self.make_experience_flop / 1e12 / (avg_make_experience_duration + 1e-12)

        avg_learn_throughput = self.learn_num_samples / (avg_learn_duration + 1e-12)
        avg_learn_tflops = self.learn_flop / 1e12 / (avg_learn_duration + 1e-12)

        print_rank_0(
            f'Making experience throughput: {avg_make_experience_throughput:.3f} samples/sec, TFLOPS: {avg_make_experience_tflops:.3f}'
        )
        print_rank_0(f'Learning throughput: {avg_learn_throughput:.3f} samples/sec, TFLOPS: {avg_learn_tflops:.3f}')

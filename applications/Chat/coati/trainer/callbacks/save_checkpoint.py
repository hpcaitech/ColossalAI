import os

import torch.distributed as dist
from coati.trainer.strategies import GeminiStrategy, LowLevelZeroStrategy, Strategy
from coati.trainer.utils import is_rank_0
from torch import nn
from torch.optim import Optimizer

from .base import Callback


class SaveCheckpoint(Callback):
    """
        The callback for saving checkpoint for coati.

        Only support saving actor and critic model.
        A typical architecture of the saved checkpoint would be:
            - checkpoint
                - episode_x
                    - actor.pt
                    - actor-optim-rank-0.pt
                    - actor-optim-rank-1.pt
                    - critic.pt
                    - critic-optim-rank-0.pt
                    - critic-optim-rank-1.pt
                - ...

    Args:
        path(str): the base path you want to save checkpoint, the checkpoint would be saved at `path/checkpoint`
        interval(int): the interval episode of saving checkpoint
        strategy(Strategy): the strategy used to train
        actor(nn.Module): the actor model
        critic(nn.Module): the critic model
        actor_optim(Optimizer): the optimizer of actor
        critic_optim(Optimizer): the optimizer of critic

    """

    def __init__(self,
                 path: str,
                 interval: int,
                 strategy: Strategy,
                 actor: nn.Module = None,
                 critic: nn.Module = None,
                 actor_optim: Optimizer = None,
                 critic_optim: Optimizer = None) -> None:
        super().__init__()
        self.path = os.path.join(path, 'checkpoint')
        self.interval = interval
        self.strategy = strategy
        self.model_dict = {'actor': [actor, actor_optim], 'critic': [critic, critic_optim]}

    def on_episode_end(self, episode: int) -> None:
        if (episode + 1) % self.interval != 0:
            return
        base_path = os.path.join(self.path, f'episode_{episode}')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for model in self.model_dict.keys():

            # save model
            if self.model_dict[model][0] is None:
                # saving only optimizer states is meaningless, so it would be skipped
                continue
            model_path = os.path.join(base_path, f'{model}.pt')
            self.strategy.save_model(model=self.model_dict[model][0], path=model_path, only_rank0=True)

            # save optimizer
            if self.model_dict[model][1] is None:
                continue
            only_rank0 = not isinstance(self.strategy, (LowLevelZeroStrategy, GeminiStrategy))
            rank = 0 if is_rank_0() else dist.get_rank()
            optim_path = os.path.join(base_path, f'{model}-optim-rank-{rank}.pt')
            self.strategy.save_optimizer(optimizer=self.model_dict[model][1], path=optim_path, only_rank0=only_rank0)

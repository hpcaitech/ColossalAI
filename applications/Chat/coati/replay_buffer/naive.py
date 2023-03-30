import random
from typing import List

import torch
from chatgpt.experience_maker.base import Experience

from .base import ReplayBuffer
from .utils import BufferItem, make_experience_batch, split_experience_batch


class NaiveReplayBuffer(ReplayBuffer):
    """Naive replay buffer class. It stores experience.

     Args:
         sample_batch_size (int): Batch size when sampling.
         limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
         cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True) -> None:
        super().__init__(sample_batch_size, limit)
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        # TODO(ver217): add prefetch
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device('cpu'))
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch)
        return experience

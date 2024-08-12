from typing import List

import torch
from coati.experience_buffer.utils import BufferItem, make_experience_batch, split_experience_batch
from coati.experience_maker.base import Experience

# from torch.multiprocessing import Queue
from ray.util.queue import Queue


class DetachedReplayBuffer:
    """
        Detached replay buffer. Share Experience across workers on the same node.
        Therefore, a trainer node is expected to have only one instance.
        It is ExperienceMakerHolder's duty to call append(exp) method, remotely.

    Args:
        sample_batch_size: Batch size when sampling. Exp won't enqueue until they formed a batch.
        tp_world_size: Number of workers in the same tp group
        limit: Limit of number of experience sample BATCHs. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload: Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0) -> None:
        self.sample_batch_size = sample_batch_size
        self.limit = limit
        self.items = Queue(self.limit, actor_options={"num_cpus": 1})
        self.batch_collector: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        """
        Expected to be called remotely.
        """
        items = split_experience_batch(experience)
        self.extend(items)

    @torch.no_grad()
    def extend(self, items: List[BufferItem]) -> None:
        """
        Expected to be called remotely.
        """
        self.batch_collector.extend(items)
        while len(self.batch_collector) >= self.sample_batch_size:
            items = self.batch_collector[: self.sample_batch_size]
            experience = make_experience_batch(items)
            self.items.put(experience, block=True)
            self.batch_collector = self.batch_collector[self.sample_batch_size :]

    def clear(self) -> None:
        # self.items.close()
        self.items.shutdown()
        self.items = Queue(self.limit)
        self.worker_state = [False] * self.tp_world_size
        self.batch_collector = []

    @torch.no_grad()
    def sample(self, worker_rank=0, to_device="cpu") -> Experience:
        ret = self._sample_and_erase()
        ret.to_device(to_device)
        return ret

    @torch.no_grad()
    def _sample_and_erase(self) -> Experience:
        ret = self.items.get(block=True)
        return ret

    def get_length(self) -> int:
        ret = self.items.qsize()
        return ret

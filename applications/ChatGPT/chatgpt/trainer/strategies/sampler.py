import math

import numpy as np


class DistributedSampler:

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        if len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas    # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

        indices = list(range(len(self.dataset)))
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.indices = indices

    def sample(self, batch_size: int) -> list:
        sampled_indices = np.random.choice(self.indices, batch_size, replace=False)
        return [self.dataset[idx] for idx in sampled_indices]

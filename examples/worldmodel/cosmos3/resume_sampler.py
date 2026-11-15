"""Samplers that can resume partway through an epoch.

Resuming at an optimizer step must not re-decode the samples that step already consumed.
``DistributedSampler`` yields a deterministic list of indices, so slicing that list by whole
batches skips the consumed work without touching ``Dataset.__getitem__``.  The slice is taken
from this rank's own stream, so the per-rank assignment -- and therefore the global batch --
is unchanged.
"""

from __future__ import annotations

import torch


class ResumeDistributedSampler(torch.utils.data.DistributedSampler):
    """``DistributedSampler`` that drops the first ``start_batch`` batches of the epoch."""

    def __init__(self, *args, start_batch: int = 0, batch_size: int = 1, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_batch = int(start_batch)
        self.batch_size = int(batch_size)
        if int(start_epoch) != self.epoch:
            self.epoch = int(start_epoch)

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            # A new epoch starts at its first batch, whatever the resumed epoch did. Setting the
            # same epoch again (the resume path) must keep the offset.
            self.start_batch = 0
        super().set_epoch(epoch)

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_batch * self.batch_size :])

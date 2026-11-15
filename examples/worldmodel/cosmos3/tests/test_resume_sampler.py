"""The resumable sampler must skip consumed batches without touching the dataset."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from examples.worldmodel.cosmos3.resume_sampler import ResumeDistributedSampler


class CountingDataset(Dataset):
    """Yields its index and records how many times an item was materialized."""

    def __init__(self, size):
        self.size = size
        self.reads = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads.append(index)
        return index


def build(dataset, *, start_batch, batch_size, rank=0, world=1, epoch=0):
    sampler = ResumeDistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=False,
        drop_last=True,
        start_batch=start_batch,
        batch_size=batch_size,
        start_epoch=epoch,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)


def batches(loader):
    return [[int(value) for value in batch] for batch in loader]


def test_resume_matches_the_tail_and_never_reads_skipped_samples():
    dataset = CountingDataset(23)
    full = batches(build(dataset, start_batch=0, batch_size=2))
    resumed_dataset = CountingDataset(23)
    resumed = batches(build(resumed_dataset, start_batch=3, batch_size=2))
    assert resumed == full[3:]
    # Exactly the tail was decoded; the skipped batches were never read.
    assert resumed_dataset.reads == [index for batch in full[3:] for index in batch]
    assert len(dataset.reads) == len(resumed_dataset.reads) + 3 * 2


def test_offset_is_per_rank_and_keeps_the_global_batch():
    world = 2
    full = [batches(build(CountingDataset(23), start_batch=0, batch_size=2, rank=rank, world=world)) for rank in range(world)]
    resumed = [
        batches(build(CountingDataset(23), start_batch=2, batch_size=2, rank=rank, world=world))
        for rank in range(world)
    ]
    for rank in range(world):
        assert resumed[rank] == full[rank][2:]
    # Peers keep disjoint samples, so the global batch composition is unchanged.
    assert set(full[0][0]).isdisjoint(full[1][0])


def test_new_epoch_restarts_at_the_first_batch():
    loader = build(CountingDataset(10), start_batch=2, batch_size=2)
    assert batches(loader)[0] == [4, 5]
    loader.sampler.set_epoch(1)
    assert batches(loader)[0] == [0, 1]


def test_resume_at_the_epoch_boundary_yields_an_empty_stream():
    loader = build(CountingDataset(10), start_batch=5, batch_size=2)
    # 10 // 1 replica // 2 per batch == 5 batches; resuming at 5 means the epoch is done.
    assert batches(loader) == []
    loader.sampler.set_epoch(1)
    assert batches(loader) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_same_epoch_keeps_the_offset():
    loader = build(CountingDataset(10), start_batch=3, batch_size=2)
    loader.sampler.set_epoch(0)
    assert batches(loader) == [[6, 7], [8, 9]]


def test_length_still_describes_a_whole_epoch():
    loader = build(CountingDataset(10), start_batch=3, batch_size=2)
    # BatchSampler derives its length from the sampler's, which describes the full epoch; the
    # training loop only uses next(), and an empty iterator is handled as a rollover.
    assert len(loader) == 5

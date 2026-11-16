"""Per-visit seeding: ``(seed, epoch, index)`` decides the draw, not which worker fetched the sample."""

from __future__ import annotations

import random

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from examples.worldmodel.cosmos3.resume_sampler import ResumeDistributedSampler
from examples.worldmodel.cosmos3.runtime import VisitRandomness, visit_seed


def test_visit_seed_is_stable_and_spread():
    assert visit_seed(1, 0, 7) == visit_seed(1, 0, 7)
    assert len({visit_seed(1, 0, index) for index in range(1000)}) == 1000
    assert visit_seed(1, 1, 7) != visit_seed(1, 0, 7)
    assert visit_seed(2, 0, 7) != visit_seed(1, 0, 7)
    assert 0 <= visit_seed(1, 0, 7) < 2**63


def test_seeded_pins_both_streams_and_puts_them_back():
    torch.manual_seed(1234)
    random.seed(1234)
    torch.rand(3)
    random.random()  # the enclosing stream advances
    with VisitRandomness.seeded(99):
        inside = (torch.rand(3), random.random())
    after = (torch.rand(3), random.random())

    # The enclosing stream continues as if the visit had not consumed anything.
    torch.manual_seed(1234)
    random.seed(1234)
    torch.rand(3)
    random.random()
    reference = (torch.rand(3), random.random())
    torch.testing.assert_close(after[0], reference[0], rtol=0, atol=0)
    assert after[1] == reference[1]

    # And the visit itself is repeatable.
    with VisitRandomness.seeded(99):
        again = (torch.rand(3), random.random())
    torch.testing.assert_close(inside[0], again[0], rtol=0, atol=0)
    assert inside[1] == again[1]


class _Toy(Dataset):
    """Stands in for the real readers: each visit is two draws from the process streams."""

    def __init__(self, seed: int) -> None:
        self.visit = VisitRandomness(seed)

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        with VisitRandomness.seeded(self.visit.seed_for(index)):
            return torch.rand(4), random.random()

    def set_visit_epoch(self, epoch: int) -> None:
        self.visit.set_epoch(epoch)


def _expected_rows(base_seed: int, epoch: int):
    rows = []
    for index in range(8):
        with VisitRandomness.seeded(visit_seed(base_seed, epoch, index)):
            rows.append((torch.rand(4), random.random()))
    return rows


@pytest.mark.parametrize("workers", [0, 2])
def test_every_worker_draws_the_same_visit(workers):
    dataset = DataLoader(_Toy(seed=17), batch_size=1, shuffle=False, num_workers=workers, persistent_workers=False)
    got = [(tensor[0], value.item()) for tensor, value in dataset]
    expected_rows = _expected_rows(17, 0)
    for index, (tensor, value) in enumerate(got):
        expected_tensor, expected_value = expected_rows[index]
        torch.testing.assert_close(tensor, expected_tensor, rtol=0, atol=0)
        assert value == expected_value


def test_the_epoch_reaches_the_workers_through_the_sampler():
    dataset = _Toy(seed=17)
    sampler = ResumeDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, batch_size=1, start_epoch=2)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=2)
    first = [tensor[0] for tensor, _ in loader]
    sampler.set_epoch(3)
    second = [tensor[0] for tensor, _ in loader]
    for index in range(8):
        torch.testing.assert_close(first[index], _expected_rows(17, 2)[index][0], rtol=0, atol=0)
        torch.testing.assert_close(second[index], _expected_rows(17, 3)[index][0], rtol=0, atol=0)


class _EpochSink(Dataset):
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return index

    def set_visit_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def test_the_sampler_pushes_its_epoch_into_the_dataset():
    dataset = _EpochSink()
    sampler = ResumeDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, batch_size=1, start_epoch=3)
    assert dataset.epochs[-1] == 3
    sampler.set_epoch(4)
    assert dataset.epochs[-1] == 4
    sampler.set_epoch(4)
    assert dataset.epochs[-1] == 4
    assert sampler.start_batch == 0


def test_a_resumed_sampler_keeps_its_offset_and_still_syncs():
    dataset = _EpochSink()
    sampler = ResumeDistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=False, batch_size=2, start_epoch=2, start_batch=3
    )
    assert dataset.epochs[-1] == 2
    sampler.set_epoch(2)
    assert sampler.start_batch == 3
    assert dataset.epochs[-1] == 2
    sampler.set_epoch(5)
    assert sampler.start_batch == 0
    assert dataset.epochs[-1] == 5

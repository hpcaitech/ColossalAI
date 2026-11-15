"""Checks on the episode-level split.

The split is not ours: cosmos-framework's DROID loader draws the same permutation with
``torch.randperm(total_episodes, generator=manual_seed(42))`` and holds out the first 3%. Pinning the exact ids is
the point — a held-out number is only comparable across the two codebases when both measured the same episodes.
The totals are the ones in this release's ``meta/info.json`` (success 57,639, failure 14,268 episodes).

Runs on CPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_dataset import split_episode_ids

SUCCESS_EPISODES = 57639
FAILURE_EPISODES = 14268


def test_val_ids_match_the_upstream_permutation():
    """Values produced by upstream's split_episode_ids, transcribed rather than recomputed."""
    assert split_episode_ids(SUCCESS_EPISODES, 42, 0.03, "val")[:10] == [
        48330,
        22578,
        51250,
        19933,
        16050,
        39038,
        37597,
        38203,
        19321,
        22742,
    ]
    assert split_episode_ids(FAILURE_EPISODES, 42, 0.03, "val")[:5] == [6150, 13604, 1028, 4732, 4582]


@pytest.mark.parametrize("total", [SUCCESS_EPISODES, FAILURE_EPISODES])
def test_sizes_and_disjointness(total):
    train = split_episode_ids(total, 42, 0.03, "train")
    val = split_episode_ids(total, 42, 0.03, "val")
    full = split_episode_ids(total, 42, 0.03, "full")
    assert len(val) == round(total * 0.03)
    assert len(train) == total - len(val)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(full) == set(range(total))


def test_full_is_the_permutation_not_the_identity():
    full = split_episode_ids(1000, 42, 0.03, "full")
    assert sorted(full) == list(range(1000))
    assert full != list(range(1000))


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        split_episode_ids(100, 42, 0.03, "test")

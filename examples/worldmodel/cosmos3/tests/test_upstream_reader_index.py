"""``episode_id`` does not identify a record in the vendored reader's index, and two things make it so.

The reader's ``_episode_records`` holds one entry per *span*, not per episode. With the curated window
filter on, an episode contributes one span per kept range and its id repeats across them; and the id counts
from zero again in every registered root, so on a mixed success-and-failure run the same id names two
different episodes. A table keyed by the id alone collapses both cases.

That is quiet in the worst way: the samples themselves stay correct -- the reader resolves them -- while
every sample's reported episode and frame number comes out wrong, which is what a replay pack, a held-out
manifest and a sample id are all built from. The reader is stubbed here so the index logic is tested on its
own, without the dataset or the framework's dependencies.

Runs on CPU.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ROOT_NAME = "droid_plus_lerobot_640x360_20260412"

# One episode, id 5, split into two kept ranges by the window filter, in dataset 0; then the same id
# again in dataset 1, which is what a second registered root gives. Rows are the dataset rows each span
# starts at, so episode 5 of dataset 0 owns rows 100.. and episode 5 of dataset 1 owns rows 40...
RECORDS = [
    (0, 100, 3, 5),
    (0, 120, 2, 5),
    (1, 40, 2, 5),
]
CUM_ENDS = [3, 5, 7]


class _Reader:
    """The two attributes the adapter reads from upstream's index, and its length."""

    def __init__(self, **kwargs):
        self._episode_records = list(RECORDS)
        self._episode_cum_ends = list(CUM_ENDS)

    def __len__(self):
        return CUM_ENDS[-1]


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    module = types.ModuleType("cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset")
    module.DROIDLeRobotDataset = _Reader
    for name in (
        "cosmos_framework",
        "cosmos_framework.data",
        "cosmos_framework.data.generator",
        "cosmos_framework.data.generator.action",
        "cosmos_framework.data.generator.action.datasets",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("DROID_FILTER_SOURCE_PREFIX", "gs://bucket/data")
    from examples.worldmodel.cosmos3.upstream_dataset import UpstreamDroidDataset

    root = tmp_path / ROOT_NAME
    root.mkdir()
    return UpstreamDroidDataset(root, chunk_length=32)


def test_every_span_of_one_episode_is_found(dataset):
    """Keyed by the episode id alone, only the last of the three spans survived the table."""
    assert [dataset._locate(index) for index in range(7)] == [
        (0, 5, 0),
        (0, 5, 1),
        (0, 5, 2),
        (0, 5, 20),
        (0, 5, 21),
        (1, 5, 0),
        (1, 5, 1),
    ]


def test_the_frame_number_counts_from_the_episode_not_the_span(dataset):
    """Counted from the span, the second kept range would restart at zero and collide with the first."""
    _, _, first_range = dataset._locate(0)
    _, _, second_range = dataset._locate(3)
    assert (first_range, second_range) == (0, 20)


def test_the_same_episode_id_in_two_roots_stays_two_episodes(dataset):
    assert dataset._locate(2)[:2] == (0, 5)
    assert dataset._locate(5)[:2] == (1, 5)


def test_no_two_windows_share_an_identity(dataset):
    ids = [dataset.sample_id(index) for index in range(len(dataset))]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "success:dataset_0:episode_5:0:32"
    assert ids[3] == "success:dataset_0:episode_5:20:32"
    assert ids[5] == "success:dataset_1:episode_5:0:32"


def test_an_index_outside_the_reader_is_refused(dataset):
    """Reported against the last span instead, it would name a window that was never drawn."""
    for index in (-1, 7, 99):
        with pytest.raises(IndexError):
            dataset._locate(index)

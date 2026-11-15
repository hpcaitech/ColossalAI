"""The vendored reader only applies the curated window filter when it is told the filter's URI prefix.

``keep_ranges_1_0_1.json`` is keyed by composite ``gs://`` URIs, and the reader refuses to guess the prefix in
front of the episode id rather than select the wrong episodes silently. The adapter used to hard-code an empty
prefix, which turned ``--keep-ranges`` on that file into a ``ValueError`` on the default reader while the same
flag worked on the other one. These tests pin the forwarding, not the reader's own filtering, which upstream
tests: the reader is stubbed so this runs without the dataset or the framework's dependencies.

Measured against the real release on 2026-11-14: both readers index 16,338,993 windows unfiltered and
12,157,927 with the filter (74.41%), the share upstream's recipe documents for the released model.

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


class _Reader:
    """Records the keyword arguments the adapter forwards, and nothing else."""

    last: dict = {}

    def __init__(self, **kwargs):
        type(self).last = kwargs
        # (dataset index, first row, windows in the span, episode id), the shape upstream appends.
        self._episode_records = [(0, 0, 8, 0)]
        self._episode_cum_ends = [8]

    def __len__(self):
        return 8


@pytest.fixture
def adapter(monkeypatch):
    """The adapter with upstream's reader replaced, so no framework dependency is needed."""
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
    from examples.worldmodel.cosmos3.upstream_dataset import UpstreamDroidDataset

    return UpstreamDroidDataset


def test_an_explicit_prefix_reaches_the_reader(adapter, tmp_path, monkeypatch):
    monkeypatch.delenv("DROID_FILTER_SOURCE_PREFIX", raising=False)
    root = tmp_path / ROOT_NAME
    root.mkdir()
    adapter(root, keep_ranges="keep.json", filter_dict_source_prefix="gs://bucket/data")
    assert _Reader.last["use_filter_dict"] is True
    assert _Reader.last["filter_dict_path"] == "keep.json"
    assert _Reader.last["filter_dict_source_prefix"] == "gs://bucket/data"


def test_the_environment_supplies_the_prefix_when_the_caller_does_not(adapter, tmp_path, monkeypatch):
    """The same variable the other reader honours, so one setting covers both."""
    monkeypatch.setenv("DROID_FILTER_SOURCE_PREFIX", "gs://bucket/data")
    root = tmp_path / ROOT_NAME
    root.mkdir()
    adapter(root, keep_ranges="keep.json")
    assert _Reader.last["filter_dict_source_prefix"] == "gs://bucket/data"


def test_an_explicit_prefix_wins_over_the_environment(adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("DROID_FILTER_SOURCE_PREFIX", "gs://wrong/data")
    root = tmp_path / ROOT_NAME
    root.mkdir()
    adapter(root, keep_ranges="keep.json", filter_dict_source_prefix="gs://bucket/data")
    assert _Reader.last["filter_dict_source_prefix"] == "gs://bucket/data"


def test_no_filter_leaves_the_reader_unfiltered(adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("DROID_FILTER_SOURCE_PREFIX", "gs://bucket/data")
    root = tmp_path / ROOT_NAME
    root.mkdir()
    adapter(root)
    assert _Reader.last["use_filter_dict"] is False
    assert _Reader.last["filter_dict_path"] is None

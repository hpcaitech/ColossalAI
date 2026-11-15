"""Checks on the two DROID roots, the curated window set and its key protocol.

Neither is ours: the roots are upstream's ``LEROBOT_ROOTS`` for this layout and the ranges are
``keep_ranges_1_0_1.json`` (``KarlP/droid``). A run only reproduces the released numbers if it trains the same
episodes and the same windows, so the helpers are pinned and the dataset is exercised on a tiny synthetic LeRobot
layout. The filter keys follow cosmos-framework's ``normalize_episode_filter_keys`` protocol: relative episode ids
are used directly, legacy composite URI keys need an explicit source prefix, and a file that matches nothing is an
error rather than an empty training set.

Runs on CPU.
"""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_dataset import DroidActionPolicyDataset, episode_starts, filtered_starts, keep_ranges_key, resolve_shards

CAM_FEATURES = (
    "observation.image.wrist_image_left",
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
)

# A legacy composite URI key written out by hand, not built with keep_ranges_key: what these tests check is the
# protocol, and deriving the input with the production helper would only prove self-consistency.
LEGACY_URI = (
    "gs://xembodiment_data/r2d2/r2d2-data-full/A/success/kept/recordings/"
    "MP4--gs://xembodiment_data/r2d2/r2d2-data-full/A/success/kept/trajectory.h5"
)
SOURCE_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full"


def write_root(root: Path, episodes: list[dict]) -> None:
    """A LeRobot root with just enough metadata for the index builder; no videos, no data files."""
    (root / "meta/episodes").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_episodes": len(episodes)}))
    cols = {
        "episode_index": [e["index"] for e in episodes],
        "episode_id": [e["id"] for e in episodes],
        "tasks": [[e.get("task", "pick up the object")] for e in episodes],
        "length": [e["length"] for e in episodes],
        "data/chunk_index": [0] * len(episodes),
        "data/file_index": [0] * len(episodes),
        "dataset_from_index": [e["from"] for e in episodes],
    }
    for feature in CAM_FEATURES:
        cols[f"videos/{feature}/chunk_index"] = [0] * len(episodes)
        cols[f"videos/{feature}/file_index"] = [0] * len(episodes)
        cols[f"videos/{feature}/from_timestamp"] = [0.0] * len(episodes)
    pq.write_table(pa.table(cols), root / "meta/episodes/file-000.parquet")


# ---------- roots ----------


def test_resolve_shards_reads_both_roots_in_upstream_order(tmp_path):
    write_root(tmp_path / "success", [])
    write_root(tmp_path / "failure", [])
    assert [name for name, _ in resolve_shards(tmp_path)] == ["success", "failure"]


def test_resolve_shards_requires_the_parent_of_the_roots(tmp_path):
    write_root(tmp_path / "success", [])
    with pytest.raises(FileNotFoundError, match="parent"):
        resolve_shards(tmp_path / "success")


def test_success_only_drops_the_failure_root(tmp_path):
    write_root(tmp_path / "success", [])
    write_root(tmp_path / "failure", [])
    assert [name for name, _ in resolve_shards(tmp_path, success_only=True)] == ["success"]


def test_resolve_shards_rejects_something_that_is_not_a_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_shards(tmp_path)


# ---------- curated window set ----------


def test_keep_ranges_key_matches_the_upstream_format():
    episode = "AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023"
    assert keep_ranges_key(episode) == (
        "gs://xembodiment_data/r2d2/r2d2-data-full/AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023/recordings/"
        "MP4--gs://xembodiment_data/r2d2/r2d2-data-full/AUTOLab/success/2023-07-07/Fri_Jul__7_09:42:23_2023/"
        "trajectory.h5"
    )


def test_relative_episode_ids_are_used_directly(tmp_path):
    """The protocol's primary form: a complete relative episode id is the key."""
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({"A/success/kept": [[0, 50]]}))
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", keep_ranges=keep)
    assert [window.episode_id for window in dataset.windows] == ["A/success/kept"] * 18
    assert dataset.filter_stats == {"matched_episodes": 1, "kept_windows": 18}


def test_legacy_uri_keys_need_an_explicit_source_prefix(tmp_path):
    """Guessing the prefix would silently match the wrong episodes, so a URI key without one is rejected."""
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({LEGACY_URI: [[0, 50]]}))
    with pytest.raises(ValueError, match="source_prefix"):
        DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", keep_ranges=keep)


def test_legacy_uri_keys_are_normalized_with_the_source_prefix(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({LEGACY_URI: [[0, 50]]}))
    dataset = DroidActionPolicyDataset(
        tmp_path,
        chunk_length=32,
        split="full",
        keep_ranges=keep,
        filter_dict_source_prefix=SOURCE_PREFIX,
    )
    assert [window.episode_id for window in dataset.windows] == ["A/success/kept"] * 18


def test_the_source_prefix_environment_variable_is_the_default(tmp_path, monkeypatch):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({LEGACY_URI: [[0, 50]]}))
    monkeypatch.setenv("DROID_FILTER_SOURCE_PREFIX", SOURCE_PREFIX)
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", keep_ranges=keep)
    assert [window.episode_id for window in dataset.windows] == ["A/success/kept"] * 18


def test_a_uri_prefix_that_does_not_match_is_rejected(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({LEGACY_URI: [[0, 50]]}))
    with pytest.raises(ValueError, match="does not match"):
        DroidActionPolicyDataset(
            tmp_path,
            chunk_length=32,
            split="full",
            keep_ranges=keep,
            filter_dict_source_prefix="gs://some/other/prefix",
        )


def test_duplicate_entries_with_conflicting_ranges_raise(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({"A/success/kept": [[0, 50]], LEGACY_URI: [[10, 50]]}))
    with pytest.raises(ValueError, match="Conflicting"):
        DroidActionPolicyDataset(
            tmp_path,
            chunk_length=32,
            split="full",
            keep_ranges=keep,
            filter_dict_source_prefix=SOURCE_PREFIX,
        )


def test_a_filter_that_matches_no_episode_raises(tmp_path):
    """A non-empty file that selects nothing is a misconfiguration, not an empty training run."""
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({"A/success/absent": [[0, 50]]}))
    with pytest.raises(ValueError, match="matches none"):
        DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", keep_ranges=keep)


def test_episode_starts_counts_frames_minus_chunk():
    assert episode_starts(33, 32) == [0]
    assert episode_starts(36, 32) == [0, 1, 2, 3]
    assert episode_starts(32, 32) == []  # not one whole window in the episode


def test_episode_starts_follows_the_stride():
    assert episode_starts(36, 32, stride=2) == [0, 2]


def test_a_covering_range_matches_the_unfiltered_index():
    assert filtered_starts([[0, 10_000]], length=100, chunk_length=32) == episode_starts(100, 32)


def test_a_range_keeps_the_starts_that_end_inside_it():
    """A window needs 33 frames, so the last start a [40, 80) range can hold is 47."""
    assert filtered_starts([[40, 80]], length=100, chunk_length=32) == list(range(40, 48))


def test_a_range_shorter_than_a_window_keeps_nothing():
    assert filtered_starts([[10, 20]], length=100, chunk_length=32) == []


def test_a_range_is_clamped_to_the_episode():
    assert filtered_starts([[0, 10_000]], length=36, chunk_length=32) == [0, 1, 2, 3]


def test_filtered_stride_uses_frame_coordinates_and_range_origin():
    assert filtered_starts([[20, 60]], length=100, chunk_length=32, stride=2) == [20, 22, 24, 26]
    assert filtered_starts([[21, 60]], length=100, chunk_length=32, stride=3) == [21, 24, 27]
    assert filtered_starts([[-20, 999]], length=39, chunk_length=32, stride=3) == [0, 3, 6]


@pytest.mark.parametrize("stride", [0, -1])
def test_window_helpers_reject_nonpositive_stride(stride):
    with pytest.raises(ValueError, match="stride"):
        episode_starts(100, 32, stride)
    with pytest.raises(ValueError, match="stride"):
        filtered_starts([[20, 60]], 100, 32, stride)


def test_repeated_ranges_are_not_deduped():
    """Upstream appends per range without merging; deduping here would train a different number of windows."""
    assert filtered_starts([[0, 50], [0, 50]], length=60, chunk_length=32) == list(range(18)) * 2


# ---------- the dataset over both roots ----------


def test_the_dataset_indexes_both_roots(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/x", "length": 40, "from": 0}])
    write_root(tmp_path / "failure", [{"index": 0, "id": "B/failure/y", "length": 40, "from": 0}])
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full")
    assert sorted(dataset.split_ids) == ["failure", "success"]
    assert {window.root.name for window in dataset.windows} == {"success", "failure"}
    assert len(dataset) == 16  # 8 starts per episode


def test_success_only_indexes_one_root(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/success/x", "length": 40, "from": 0}])
    write_root(tmp_path / "failure", [{"index": 0, "id": "B/failure/y", "length": 40, "from": 0}])
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", success_only=True)
    assert list(dataset.split_ids) == ["success"]
    assert {window.root.name for window in dataset.windows} == {"success"}


def test_keep_ranges_drops_unlisted_episodes_and_windows(tmp_path):
    write_root(
        tmp_path / "success",
        [
            {"index": 0, "id": "A/success/kept", "length": 100, "from": 0},
            {"index": 1, "id": "A/success/absent", "length": 100, "from": 100},
        ],
    )
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({"A/success/kept": [[0, 50]]}))
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", keep_ranges=keep)
    assert [window.episode_id for window in dataset.windows] == ["A/success/kept"] * 18


def test_max_episodes_counts_episodes_the_filter_keeps(tmp_path):
    """max_episodes has to truncate inside the kept set, not leave a smoke run with nothing to train on."""
    write_root(
        tmp_path / "success",
        [{"index": index, "id": f"A/success/{index}", "length": 100, "from": index * 100} for index in range(3)],
    )
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({"A/success/2": [[0, 50]]}))
    dataset = DroidActionPolicyDataset(
        tmp_path, chunk_length=32, split="full", max_episodes=1, episode_shuffle_seed=42, keep_ranges=keep
    )
    assert [window.episode_id for window in dataset.windows] == ["A/success/2"] * 18

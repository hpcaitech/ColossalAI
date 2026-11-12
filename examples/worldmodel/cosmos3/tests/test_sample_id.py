"""The logical identity of a training window is path-independent, per-root and per-window-length.

The id ties a GPU sample to an NPU sample, a held-out manifest entry to a resumed run, and a replay pack to the
window it was recorded from, so it must not embed a mount path and must not collapse two different windows into
one. Repeated draws of one window (overlapping curated ranges) keep the id and differ only by ``draw_id``. The
first ten constructor arguments also keep the positional order they had before the multi-root extension.

Runs on CPU.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_dataset import DroidActionPolicyDataset
from test_keep_ranges import write_root


def test_the_id_does_not_depend_on_where_the_release_is_mounted(tmp_path):
    episodes = [{"index": 0, "id": "A/success/ep", "length": 40, "from": 0}]
    write_root(tmp_path / "h20" / "success", episodes)
    write_root(tmp_path / "npu" / "success", episodes)
    gpu = DroidActionPolicyDataset(tmp_path / "h20", chunk_length=32, split="full")
    npu = DroidActionPolicyDataset(tmp_path / "npu", chunk_length=32, split="full")
    gpu_ids = [gpu.sample_id(index) for index in range(len(gpu))]
    assert gpu_ids == [npu.sample_id(index) for index in range(len(npu))]
    assert all(str(tmp_path) not in sample_id for sample_id in gpu_ids)


def test_the_same_episode_id_in_two_roots_does_not_collide(tmp_path):
    episodes = [{"index": 0, "id": "A/same", "length": 40, "from": 0}]
    write_root(tmp_path / "success", episodes)
    write_root(tmp_path / "failure", episodes)
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full")
    ids = [dataset.sample_id(index) for index in range(len(dataset))]
    assert len(ids) == 16
    assert len(set(ids)) == 16
    assert any(sample_id.startswith("success:A/same:") for sample_id in ids)
    assert any(sample_id.startswith("failure:A/same:") for sample_id in ids)


def test_the_window_length_is_part_of_the_id(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/ep", "length": 100, "from": 0}])
    short = DroidActionPolicyDataset(tmp_path, chunk_length=16, split="full")
    long = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full")
    assert short.sample_id(0) == "success:A/ep:0:16"
    assert long.sample_id(0) == "success:A/ep:0:32"
    assert short.sample_id(0) != long.sample_id(0)


def test_the_dataset_revision_prefixes_the_id(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/ep", "length": 40, "from": 0}])
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", dataset_revision="v1.0.1")
    assert dataset.sample_id(0) == "v1.0.1:success:A/ep:0:32"


def test_overlapping_ranges_share_one_id_and_add_a_draw_id(tmp_path):
    write_root(tmp_path / "success", [{"index": 0, "id": "A/kept", "length": 100, "from": 0}])
    keep = tmp_path / "keep_ranges.json"
    keep.write_text(json.dumps({"A/kept": [[0, 50], [0, 50]]}))
    dataset = DroidActionPolicyDataset(tmp_path, chunk_length=32, split="full", keep_ranges=keep)
    assert [window.draw_id for window in dataset.windows] == [0] * 18 + [1] * 18
    ids = [dataset.sample_id(index) for index in range(len(dataset))]
    assert ids[:18] == ids[18:]
    assert len(set(ids)) == 18


def test_the_original_positional_arguments_keep_their_meaning(tmp_path):
    """Before the multi-root extension the eighth positional argument was ``augment``; a caller that still passes
    eight or ten arguments must keep its behaviour instead of silently becoming ``split=True``."""
    write_root(tmp_path / "success", [{"index": 0, "id": "A/ep", "length": 40, "from": 0}])
    dataset = DroidActionPolicyDataset(tmp_path, 32, 15.0, "480", 1, 42, None, True)
    assert dataset.augment is True
    assert dataset.chunk_length == 32
    assert dataset.split == "train"
    without = DroidActionPolicyDataset(tmp_path, 32, 15.0, "480", 1, 42, None, False)
    assert without.augment is False
    full = DroidActionPolicyDataset(tmp_path, 32, 15.0, "480", 1, 42, None, False, "full", 0.03)
    assert full.split == "full"

"""The collate carries every field the dataset produced.

It used to carry a written-out list of nine keys. ``__getitem__`` grew two more -- ``idle_frames`` and
``additional_view_description`` -- and the list did not, so both arrived at ``build_action_json_prompt`` as
None. That function treats them as optional, so the prompt lost the multi-view description and the idle-frame
count without any error: 104 tokens where upstream, on the same window, emits 149.

So the check is not "these two keys are carried" but "no key is dropped", which is the property the written-out
list failed to have.

Runs on CPU in milliseconds; no dataset, no video decoding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_dataset import collate


def sample(index: int = 0) -> dict:
    """One sample with the same key set __getitem__ returns, tensors and non-tensors mixed."""
    return {
        "video": torch.zeros(3, 5, 8, 8),
        "action": torch.zeros(4, 8),
        "state": torch.zeros(8),
        "task": f"task {index}",
        "idle_frames": index,
        "additional_view_description": "three views",
        "episode_index": index,
        "start": 0,
        "canvas_hw": torch.tensor([544, 736]),
        "content_hw": torch.tensor([540, 640]),
        "aspect_key": "4,3",
    }


def test_no_field_is_dropped():
    one = sample()
    out = collate([one])
    assert set(out) == set(one)


def test_tensors_are_stacked_and_the_rest_stay_lists():
    out = collate([sample(0), sample(1)])
    assert out["video"].shape == (2, 3, 5, 8, 8)
    assert out["content_hw"].shape == (2, 2)
    assert out["task"] == ["task 0", "task 1"]
    assert out["idle_frames"] == [0, 1]
    assert out["additional_view_description"] == ["three views", "three views"]


def test_a_sample_missing_a_field_is_refused():
    short = sample(1)
    del short["idle_frames"]
    with pytest.raises(ValueError, match="same keys"):
        collate([sample(0), short])

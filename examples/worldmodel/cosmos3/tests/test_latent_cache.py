"""The latent cache reader must key by the dataset's logical sample id and keep its geometry."""

import json
from types import SimpleNamespace

import pytest
import torch

from examples.worldmodel.cosmos3.latent_cache import LatentCacheDataset


class BaseDataset:
    """The parts of ``DroidActionPolicyDataset`` the cache reader uses."""

    fps = 15.0

    def __init__(self, sample_ids):
        self._sample_ids = list(sample_ids)
        self.windows = [
            SimpleNamespace(episode_id="episode", start=index, task="pick", episode_index=0)
            for index in range(len(self._sample_ids))
        ]

    def __len__(self):
        return len(self.windows)

    def sample_id(self, index):
        return self._sample_ids[index]

    def _read_actions(self, window):
        return torch.zeros(32, 8), torch.zeros(8)


def write_cache(root, sample_ids):
    shard = {
        sample_id: {
            "latent": torch.full((1, 2, 2, 2, 2), float(index)),
            "canvas_hw": [32, 32],
            "content_hw": [16, 16],
            "aspect": "1,1",
        }
        for index, sample_id in enumerate(sample_ids)
    }
    torch.save(shard, root / "shard-000-0000.pt")
    index = {
        sample_id: {
            "shard": "shard-000-0000.pt",
            "shape": [1, 2, 2, 2, 2],
            "keep_hw": [2, 2],
            "canvas_hw": [32, 32],
            "aspect": "1,1",
        }
        for sample_id in sample_ids
    }
    (root / "index-rank-000.json").write_text(json.dumps(index))


def test_the_cache_is_keyed_by_the_dataset_sample_id(tmp_path):
    sample_ids = ["rev:success:episode:1:32", "rev:success:episode:2:32"]
    write_cache(tmp_path, sample_ids)

    dataset = LatentCacheDataset(BaseDataset(sample_ids), tmp_path)
    sample = dataset[1]

    assert sample["sample_id"] == sample_ids[1]
    assert sample["latent"].shape == (2, 2, 2, 2)  # the extractor's leading batch dim is dropped
    assert sample["task"] == "pick"
    torch.testing.assert_close(sample["canvas_hw"], torch.tensor([32, 32]))
    torch.testing.assert_close(sample["content_hw"], torch.tensor([16, 16]))
    assert sample["aspect_key"] == "1,1"


def test_a_partial_cache_fails_at_construction(tmp_path):
    write_cache(tmp_path, ["only-this-one"])

    with pytest.raises(KeyError, match="missing from"):
        LatentCacheDataset(BaseDataset(["only-this-one", "not-in-the-cache"]), tmp_path)

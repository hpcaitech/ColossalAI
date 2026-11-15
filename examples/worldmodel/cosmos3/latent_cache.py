"""Read pre-extracted VAE latents back as training samples.

``precompute_latents.py`` writes ``shard-<rank>-<n>.pt`` files plus one ``index-rank-<rank>.json`` per rank, keyed
by the same ``sample_id`` string the dataset produces. This module turns that directory into a map-style dataset
whose samples differ from the online one in exactly one field: ``latent`` replaces ``video``.

Why it is safe to reuse everything else: the cache stores the latent *already normalised and cropped to the content
region*, which is what ``pack_one`` consumes, and it stores the geometry (``canvas_hw``, ``content_hw``,
``aspect``) alongside it. Those geometry fields are augmentation-invariant -- the Compose crops and resizes to the
same size it was handed, so the composite and therefore the canvas do not move -- which is why a latent written
from one augmentation draw can be paired with a fresh text draw here without breaking the prompt's positions.

What the cache does *not* carry, and what therefore still comes live from the dataset: the action rows, the
initial state, the annotation draw and the idle-frame count. Those are cheap (a parquet slice) and keeping them
live means this reader cannot drift from the trainer on the parts that decide the target.

    dataset = LatentCacheDataset(DroidActionPolicyDataset(root, augment=False, ...), cache_dir)
    sample = dataset[0]            # has "latent", no "video"
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from examples.worldmodel.cosmos3.droid_dataset import (
    CONCAT_VIEW_DESCRIPTION,
    DroidActionPolicyDataset,
    count_idle_frames,
)

__all__ = ["LatentCache", "LatentCacheDataset", "load_cache_index"]


def load_cache_index(cache_dir: str | Path) -> dict[str, dict]:
    """Merge every index file in a cache directory into one ``sample_id -> entry`` map.

    The extractor writes one index per rank, because a rank only knows the samples it encoded. A single
    ``index.json`` from an older run is accepted too, so an existing directory does not have to be rebuilt.

    Args:
        cache_dir (str | Path): Directory holding ``index-rank-*.json`` (and/or ``index.json``) and shards.

    Returns:
        dict[str, dict]: ``sample_id -> {"shard", "shape", "keep_hw", "canvas_hw", "aspect"}``.

    Raises:
        FileNotFoundError: If the directory holds no index at all.
    """
    cache_dir = Path(cache_dir)
    merged: dict[str, dict] = {}
    files = sorted(cache_dir.glob("index-rank-*.json"))
    legacy = cache_dir / "index.json"
    if not files and legacy.is_file():
        files = [legacy]
    if not files:
        raise FileNotFoundError(
            f"{cache_dir} holds no index-rank-*.json or index.json; it is not a latent cache written by "
            f"precompute_latents.py."
        )
    for path in files:
        with open(path) as handle:
            merged.update(json.load(handle))
    return merged


class LatentCache:
    """Shard-backed lookup with a small shard cache.

    One shard is a few hundred MiB, so loading is cached rather than repeated: with block-ordered extraction a
    worker walks a narrow range of shards, and a two-shard window covers that walk without holding the whole cache.
    """

    def __init__(self, cache_dir: str | Path, max_cached_shards: int = 2):
        self.cache_dir = Path(cache_dir)
        self.index = load_cache_index(self.cache_dir)
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._shards: dict[str, dict] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.index

    def _load(self, shard: str) -> dict:
        blob = self._shards.get(shard)
        if blob is not None:
            return blob
        with open(self.cache_dir / shard, "rb") as handle:
            blob = torch.load(handle, map_location="cpu", weights_only=False)
        self._shards[shard] = blob
        self._order.append(shard)
        while len(self._order) > self.max_cached_shards:
            self._shards.pop(self._order.pop(0), None)
        return blob

    def get(self, sample_id: str) -> dict:
        """Cached payload for one sample.

        Raises:
            KeyError: If the sample was never extracted, with the id so a partial cache is obvious rather than a
                silent skip that changes the epoch length.
        """
        entry = self.index.get(sample_id)
        if entry is None:
            raise KeyError(f"{sample_id!r} is not in {self.cache_dir}; the cache does not cover this window.")
        return self._load(entry["shard"])[sample_id]


class LatentCacheDataset(Dataset):
    """The dataset's windows and targets, with the cached latent standing in for the decoded video.

    Args:
        base (DroidActionPolicyDataset): Supplies the window index and the action/state parquet reads. It is not
            asked for frames, so its ``augment`` setting is irrelevant here.
        cache_dir (str | Path): Directory written by ``precompute_latents.py``.
        max_cached_shards (int, optional): Shards held open per process. Defaults to 2.
    """

    def __init__(self, base: DroidActionPolicyDataset, cache_dir: str | Path, max_cached_shards: int = 2):
        self.base = base
        self.cache = LatentCache(cache_dir, max_cached_shards=max_cached_shards)
        # Check every window, not a sample of them. The sampler shuffles over the whole selection, so a gap
        # anywhere turns into a KeyError hours into a run; `len(dataset)` lookups against a dict cost nothing and
        # make a partial cache fail at startup instead.
        missing = [base.sample_id(index) for index in range(len(base)) if base.sample_id(index) not in self.cache]
        if missing:
            raise KeyError(
                f"{len(missing)} of {len(base)} windows are missing from {cache_dir}, e.g. {missing[0]!r}. "
                f"The cache was built from a different episode count, limit or window selection; re-extract with "
                f"--max-episodes/--limit covering this dataset."
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        window = self.base.windows[index]
        # The extractor keys entries by the dataset's logical sample id (revision, root role, episode, start,
        # chunk length); reconstructing it here from the physical path would miss a cache built with a revision
        # tag and could not distinguish chunk lengths.
        sample_id = self.base.sample_id(index)
        cached = self.cache.get(sample_id)
        action, state = self.base._read_actions(window)
        task = random.choice(window.task.split(" | "))
        full_action = torch.cat([state.reshape(1, -1).to(action.dtype), action], dim=0)
        return {
            # pack_one wants a [B, C, T, H, W] latent; the extractor stores each sample with its own leading batch
            # dim as [1, C, T, h, w]. Dropping it here is what lets collate stack a batch back into
            # [B, C, T, h, w] -- leaving it produces a 6-D tensor and pack_one's shape unpack fails.
            "latent": cached["latent"][0],
            "action": action,
            "state": state,
            "task": task,
            "idle_frames": count_idle_frames(full_action, self.base.fps),
            "additional_view_description": CONCAT_VIEW_DESCRIPTION,
            "episode_index": window.episode_index,
            "sample_id": sample_id,
            "start": window.start,
            "canvas_hw": torch.tensor(cached["canvas_hw"]),
            "content_hw": torch.tensor(cached["content_hw"]),
            "aspect_key": cached["aspect"],
        }

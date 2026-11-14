# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared LeRobot adapter utilities for Action datasets.

These helpers centralize common behavior across Action wrappers:
- deterministic train/val episode splitting
- valid per-episode index range construction
- a reusable BaseActionLeRobotDataset class with lazy init, video formatting,
  and common result building
"""

from __future__ import annotations

import importlib
import logging as _logging
import math
import os as _os
import random
from bisect import bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import huggingface_hub.constants as _hf_const
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

_hf_offline_applied = False


def _ensure_hf_hub_offline() -> None:
    """Force HF Hub into offline mode for local-only datasets (repo_id="local").

    Sets the ``HF_HUB_OFFLINE`` env var (for any future imports in worker
    processes), patches the already-imported constant, and suppresses the
    expected "Returning existing local_dir" fallback warning.

    Safe to call multiple times; only applies once per process.
    """
    global _hf_offline_applied
    if _hf_offline_applied:
        return
    if "HF_HUB_OFFLINE" not in _os.environ:
        _os.environ["HF_HUB_OFFLINE"] = "1"
    if not _hf_const.HF_HUB_OFFLINE:
        _hf_const.HF_HUB_OFFLINE = True
    _logging.getLogger("huggingface_hub._snapshot_download").setLevel(_logging.ERROR)
    _hf_offline_applied = True


from functools import cached_property

from cosmos_framework.data.generator.action.utils.action_processing import (
    ActionNormalizationMethod,
    ActionNormalizer,
    load_action_stats,
    resolve_action_normalization,
)

# Re-export the action_spec DSL from this module so that subclass datasets
# only need a single import block (alongside ``BaseActionLeRobotDataset``).
from cosmos_framework.data.generator.action.utils.action_spec import (  # noqa: F401  (re-export)
    ActionSpec,
    DimType,
    Gripper,
    Joint,
    Pos,
    Reserved,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.pose_utils import compute_idle_frames
from cosmos_framework.data.generator.action.utils.viewpoint_utils import Viewpoint
from cosmos_framework.utils import log


# No-op memory-profiling shims. Profiling is disabled in cosmos-framework, so these
# keep the ported dataset's optional RSS-tracking call sites cheap and side-effect-free
# (formerly a separate memprofile stub module).
def _memprofile_enabled() -> bool:
    return False


def _deep_size(obj, *args, **kwargs) -> int:
    return 0


def _fmt_mb(n, *args, **kwargs) -> str:
    return "n/a"


def log_worker_memory_breakdown(*args, **kwargs) -> None:
    return None


@contextmanager
def rss_tracker(*args, **kwargs):
    yield


# ---------------------------------------------------------------------------
# LRU-capped VideoDecoderCache
# ---------------------------------------------------------------------------
_LRU_VIDEO_CACHE_MAX_SIZE: int = 64
_LRU_DATASET_MAX_LOADED: int = 32
ActionNormalization = ActionNormalizationMethod
_ACTION_NORMALIZATION_CHOICES: tuple[str, ...] = ("quantile", "quantile_rot", "meanstd", "minmax")

_decoder_cache_patched = False


class _LRUVideoDecoderCache:
    """Drop-in replacement for ``lerobot.datasets.video_utils.VideoDecoderCache``
    with LRU eviction.  When the cache exceeds *max_size* entries the
    least-recently-used decoder (and its file handle) is evicted.
    """

    def __init__(self, max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_decoder(self, video_path: str) -> Any:
        if importlib.util.find_spec("torchcodec"):  # type: ignore[attr-defined]
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")

        import fsspec

        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
                self._hits += 1
                return self._cache[video_path][0]

            self._misses += 1
            file_handle = fsspec.open(video_path).__enter__()
            decoder = VideoDecoder(file_handle, seek_mode="approximate")  # type: ignore[arg-type]
            self._cache[video_path] = (decoder, file_handle)

            evicted = 0
            while len(self._cache) > self._max_size:
                _, (_, old_fh) = self._cache.popitem(last=False)
                try:
                    old_fh.close()
                except Exception:
                    pass
                evicted += 1
            self._evictions += evicted

            if evicted and self._evictions % 50 <= evicted:
                log.debug(
                    f"[VideoDecoderCache pid={_os.getpid()}] "
                    f"evicted={self._evictions} total, size={len(self._cache)}/{self._max_size}, "
                    f"hits={self._hits}, misses={self._misses}, "
                    f"hit_rate={100 * self._hits / max(1, self._hits + self._misses):.1f}%"
                )

            return decoder

    def clear(self) -> None:
        with self._lock:
            for _, file_handle in self._cache.values():
                try:
                    file_handle.close()
                except Exception:
                    pass
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


def _patch_decoder_cache(max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
    """Replace the module-level ``_default_decoder_cache`` in LeRobot with an
    LRU-capped version to prevent unbounded memory growth in workers."""
    global _decoder_cache_patched
    if _decoder_cache_patched:
        return

    import lerobot.datasets.video_utils as _vu

    lru_cache = _LRUVideoDecoderCache(max_size=max_size)
    _vu._default_decoder_cache = lru_cache
    _decoder_cache_patched = True
    log.debug(f"Patched LeRobot VideoDecoderCache with LRU max_size={max_size}")


def _parallel_map(
    fn: Callable[[Any], Any],
    items: list[Any],
    *,
    max_workers: int,
    label: str,
) -> list[Any]:
    """Thread-pool ``map`` — returns results in input order.

    Intended for IO-bound prefetch (``LeRobotDatasetMetadata`` loads,
    parquet column reads).  Preserves item-order so callers can ``zip``
    with their ``indices`` / ``roots`` list.  Skips the thread pool
    entirely when there is 0 or 1 task — avoids per-worker
    ``ThreadPoolExecutor`` setup cost and log spam under
    ``shard_across_workers=True`` where each worker typically gets
    only 1-2 shards.
    """
    if not items:
        return []
    if len(items) == 1 or max_workers <= 1:
        return [fn(items[0])] if len(items) == 1 else [fn(x) for x in items]
    log.info(f"{label}: {len(items)} tasks (workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))


def split_episode_ids(total_episodes: int, seed: int, val_ratio: float, split: str) -> list[int]:
    """Create deterministic random episode ids for train/val/full splits."""
    num_val = int(round(total_episodes * val_ratio))
    g = torch.Generator().manual_seed(seed)
    episode_ids = torch.randperm(total_episodes, generator=g).tolist()

    if split == "train":
        return episode_ids[num_val:]
    if split == "val":
        return episode_ids[:num_val]
    return episode_ids


def build_episode_spans(
    episodes: Any,
    episode_ids: Sequence[int],
    chunk_length: int,
    sample_stride: int = 1,
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Build valid episode spans for LeRobot frame queries.

    Returns:
        - episode spans as ``(episode_id, sample_start, valid_len)``
        - total valid sample count across selected episodes
        - total raw frame count across selected episodes
    """
    assert sample_stride >= 1, f"sample_stride must be >= 1, got {sample_stride}"

    dataset_from_index = list(episodes["dataset_from_index"])
    dataset_to_index = list(episodes["dataset_to_index"])
    length = list(episodes["length"])

    spans: list[tuple[int, int, int]] = []
    valid_count = 0
    sample_count = 0
    for episode_id in episode_ids:
        start = dataset_from_index[episode_id]
        stop = dataset_to_index[episode_id]
        raw_valid_len = stop - start - chunk_length
        if raw_valid_len > 0:
            valid_len = (raw_valid_len + sample_stride - 1) // sample_stride
            spans.append((episode_id, start, valid_len))
            valid_count += valid_len
        sample_count += int(length[episode_id])

    return spans, valid_count, sample_count


def _normalize_split(split: str) -> str:
    """Normalize split name to one of ``'train'``, ``'val'``, ``'full'``."""
    s = split.lower().strip()
    if s in {"val", "valid", "validation", "eval", "test"}:
        return "val"
    if s in {"train", "full"}:
        return s
    raise ValueError(f"Unsupported {split=}. Use train/val/full.")


class BaseActionLeRobotDataset(Dataset):
    """Reusable base class for Action LeRobot-backed map-style datasets.

    Subclasses typically:
    1) call ``_register_source`` to register one or more LeRobot sources
    2) implement ``__getitem__`` for dataset-specific sample parsing
    3) call ``_build_result`` to assemble the return dict
    """

    # Applied as: R_opencv = R_native @ _to_opencv
    # Subclasses override in __init__; default is identity (no correction).

    # Bundled normalization stats directory.  Stats are committed at
    # ``<_NORMALIZERS_DIR>/<embodiment>_<pose>_<rotation_format>.json`` (flat
    # layout matching the existing UMI files) and produced by
    # ``projects/cosmos3/vfm/datasets/action/compute_action_stats.py``.
    # Subclasses that need a different filename scheme can override
    # :meth:`_normalizer_filename`.
    _NORMALIZERS_DIR: ClassVar[Path] = Path(__file__).parent / "normalizers"

    def __init__(
        self,
        *,
        fps: float,
        chunk_length: int,
        split_seed: int,
        split_val_ratio: float,
        split: str,
        mode: str,
        embodiment_type: str,
        viewpoint: Viewpoint,
        pose_convention: str | None = None,
        rotation_format: str | None = None,
        action_normalization: ActionNormalization | None = None,
        apply_forward_clamp: bool = False,
        tolerance_s: float = 1e-4,
        max_loaded_datasets: int = _LRU_DATASET_MAX_LOADED,
        skip_video_loading: bool = False,
        sample_stride: int = 1,
        enable_fast_init: bool = False,
        fast_init_max_workers: int = 64,
        min_episode_length_frames: int | None = None,
    ) -> None:
        super().__init__()
        _ensure_hf_hub_offline()
        _patch_decoder_cache()
        self._memprofile = _memprofile_enabled()

        assert sample_stride >= 1, f"sample_stride must be >= 1, got {sample_stride}"
        assert fast_init_max_workers >= 1, f"fast_init_max_workers must be >= 1, got {fast_init_max_workers}"
        assert action_normalization is None or action_normalization in _ACTION_NORMALIZATION_CHOICES, (
            f"action_normalization must be None or one of {_ACTION_NORMALIZATION_CHOICES}, got {action_normalization!r}"
        )

        with rss_tracker(f"{self.__class__.__name__}.__init__", enabled=self._memprofile):
            self._fps = fps
            self._dt = 1.0 / fps
            self._chunk_length = chunk_length
            self._split_seed = split_seed
            self._split_val_ratio = split_val_ratio
            self._split = _normalize_split(split)
            self._mode = mode
            self._embodiment_type = embodiment_type
            self._viewpoint: Viewpoint = viewpoint
            self._pose_convention = pose_convention
            self._rotation_format = rotation_format
            self._action_normalizer: ActionNormalizer | None = None
            if action_normalization is not None:
                # ``apply_forward_clamp`` decides whether normalized actions are
                # clamped into [-1, 1]. It matters most on the rot6d diagonal
                # channels: for DROID, R00/R11 have a q01/q99 half-width of
                # ~7.7e-4, so they saturate at roughly 3.2 deg of per-frame
                # rotation and 10 deg/frame normalizes to about -19 unclamped.
                # Callers name the intent rather than inheriting a default.
                self._action_normalizer = resolve_action_normalization(
                    action_normalization,
                    self._load_norm_stats(action_normalization),
                    apply_forward_clamp=apply_forward_clamp,
                )
            self._tolerance_s = tolerance_s
            self._max_loaded_datasets = max_loaded_datasets
            self._skip_video_loading = skip_video_loading
            self._sample_stride = sample_stride
            self._enable_fast_init = enable_fast_init
            self._fast_init_max_workers = fast_init_max_workers
            # Optional post-filter on raw episode length. When set, episodes
            # whose raw frame count is below this threshold are dropped from
            # ``_episode_records`` after ``build_episode_spans`` runs. Lets
            # eval configs select e.g. only ``> 60s`` wall-clock episodes
            # (1800 raw frames at native 30 fps) while keeping ``chunk_length``
            # (the fetch window) at a smaller value such as 900 (60 s @ fps=15).
            # Subclasses that override ``_append_index_records`` are expected
            # to honor this attribute themselves.
            self._min_episode_length_frames: int | None = min_episode_length_frames
            self._delta_timestamps: dict[str, list[float]] = {}
            self._to_opencv: np.ndarray | dict[str, np.ndarray] = np.eye(3, dtype=np.float32)

            if pose_convention is None:
                log.warning(
                    f"{self.__class__.__name__}: pose_convention is not set. "
                    "Consider specifying 'backward_framewise' or 'backward_anchored'."
                )

            self._datasets: list[LeRobotDataset | None] = []
            self._dataset_build_args: list[dict[str, Any] | None] = []
            self._loaded_lru: OrderedDict[int, None] = OrderedDict()

            # -- Flat index structures (populated by _append_index_records) --
            # Together these two lists form a searchable map from a flat
            # global index to (dataset, row, episode, frame).  One entry per
            # episode span across *all* registered sources.
            #
            # _episode_records[i] = (ds_idx, sample_start, valid_len, episode_id)
            #   ds_idx       – which source dataset (index into _datasets)
            #   sample_start – first row of this span in that dataset's table
            #   valid_len    – number of usable frames in this span
            #   episode_id   – the episode this span belongs to
            #
            # _episode_cum_ends[i] = running total of valid_len through span i
            #   Used for O(log N) lookup via bisect_right in _resolve_index.
            self._episode_records: list[tuple[int, int, int, int]] = []
            self._episode_cum_ends: list[int] = []
            self._num_valid_indices = 0
            self._domain_id = get_domain_id(self._embodiment_type)

            # Deferred-init shard roots — a list of root paths.
            # Subclasses populate this in __init__; _register_sources()
            # reads _delta_timestamps and _tolerance_s from self (both
            # initialised above, with _delta_timestamps overridden by
            # each subclass).
            # ActionUnifiedIterableDataset.assign_worker uses len() for
            # round-robin shard distribution and _register_sources(indices)
            # for deferred loading.  When empty, shard distribution is
            # skipped (every worker iterates the full dataset).
            self._all_shard_roots: list[str] = []

    # -- public properties ---------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def split(self) -> str:
        return self._split

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = value

    @property
    def domain_id(self) -> int:
        return self._domain_id

    # -- source registration -------------------------------------------------

    def _register_source(
        self,
        *,
        delta_timestamps: dict[str, list[float]],
        tolerance_s: float,
        root: str | None = None,
        repo_id: str = "local",
        force_cache_sync: bool = False,
        download_videos: bool = False,
        video_backend: str | None = None,
        revision: str | None = None,
        dataset_label: str | None = None,
        prefetched_meta: LeRobotDatasetMetadata | None = None,
    ) -> LeRobotDatasetMetadata:
        """Register a LeRobot dataset source lazily (metadata-only at init).

        ``prefetched_meta`` lets subclasses load metadata in a thread pool
        (``LeRobotDatasetMetadata`` reads are pure I/O — ``info.json`` +
        ``episodes.parquet`` + ``tasks.parquet``) and then hand the ready
        object to the serial append-path below, which still manages the
        order-sensitive shared state (``_datasets`` / ``_dataset_build_args``
        / ``_episode_records`` / ``_episode_cum_ends``).  When ``None`` the
        caller gets the original single-threaded behavior.
        """
        label_str = f" [{dataset_label}]" if dataset_label else ""
        cls = self.__class__.__name__
        # "local" is not a valid PEP 440 version, so LeRobot's
        # is_valid_version() check skips the get_safe_version() HF API call.
        if repo_id == "local" and revision is None:
            revision = "local"

        with rss_tracker(f"{cls}{label_str} — metadata load", enabled=self._memprofile):
            if prefetched_meta is not None:
                meta = prefetched_meta
            else:
                meta = LeRobotDatasetMetadata(
                    repo_id=repo_id,
                    root=root,
                    revision=revision,
                    force_cache_sync=force_cache_sync,
                )
            ds_idx = len(self._datasets)
            self._datasets.append(None)
            self._dataset_build_args.append(
                {
                    "repo_id": repo_id,
                    "root": root,
                    "delta_timestamps": delta_timestamps,
                    "tolerance_s": tolerance_s,
                    "force_cache_sync": force_cache_sync,
                    "download_videos": download_videos,
                    "video_backend": video_backend,
                    "revision": revision,
                }
            )

        with rss_tracker(
            f"{cls}{label_str} — index records",
            enabled=self._memprofile,
            extras_fn=lambda: [
                f"episode_records so far: {len(self._episode_records)} entries, "
                f"~{_fmt_mb(_deep_size(self._episode_records) / (1024 * 1024))}",
                f"episode_cum_ends so far: {len(self._episode_cum_ends)} entries, "
                f"~{_fmt_mb(_deep_size(self._episode_cum_ends) / (1024 * 1024))}",
            ],
        ):
            self._append_index_records(meta=meta, ds_idx=ds_idx, dataset_label=dataset_label)

        return meta

    def _append_index_records(
        self,
        *,
        meta: LeRobotDatasetMetadata,
        ds_idx: int,
        dataset_label: str | None = None,
    ) -> None:
        """Populate episode split / index records from dataset metadata."""
        episode_ids = split_episode_ids(
            total_episodes=meta.total_episodes,
            seed=self._split_seed,
            val_ratio=self._split_val_ratio,
            split=self._split,
        )
        # TODO: remove once bridge training switches to a pre-filtered mirror.
        if hasattr(self, "_filter_valid_episodes"):
            episode_ids = self._filter_valid_episodes(meta, episode_ids)
        episode_spans, valid_count, sample_count = build_episode_spans(
            episodes=meta.episodes,
            episode_ids=episode_ids,
            chunk_length=self._chunk_length,
            sample_stride=self._sample_stride,
        )

        # Optional duration filter (see ``self._min_episode_length_frames``
        # comment in ``__init__``). Drops episodes whose raw frame count is
        # below the threshold. Operates after ``build_episode_spans`` so the
        # threshold is decoupled from ``chunk_length`` (which controls the
        # fetch window). No-op when the attribute is None.
        if self._min_episode_length_frames is not None:
            length_lookup = list(meta.episodes["length"])
            before = len(episode_spans)
            episode_spans = [
                (eid, ss, vl)
                for (eid, ss, vl) in episode_spans
                if int(length_lookup[eid]) >= self._min_episode_length_frames
            ]
            dropped = before - len(episode_spans)
            if dropped > 0:
                log.info(
                    f"{self.__class__.__name__}: "
                    f"min_episode_length_frames={self._min_episode_length_frames} "
                    f"dropped {dropped} / {before} chunk-eligible spans"
                )

        class_name = self.__class__.__name__
        label = f" [{dataset_label}]" if dataset_label else ""
        log.info(f"{class_name}{label}: split={self._split}, num episodes={len(episode_ids)}")
        if sample_count > 0:
            log.info(
                f"{class_name}{label}: kept {valid_count} / {sample_count} "
                f"({100 * valid_count / sample_count:.2f} %) samples"
            )

        for episode_id, sample_start, valid_len in episode_spans:
            self._episode_records.append((ds_idx, sample_start, valid_len, episode_id))
            self._num_valid_indices += valid_len
            self._episode_cum_ends.append(self._num_valid_indices)

    # -- deferred shard registration -----------------------------------------

    def _register_sources(self, indices: list[int] | None = None) -> None:
        """Register a subset (or all) of the shard roots in ``_all_shard_roots``.

        Called by ``ActionUnifiedIterableDataset.assign_worker`` during training,
        or explicitly by eval/visualization scripts after construction.

        ``_all_shard_roots`` is a list of root paths.  Per-shard args that are
        shared across all shards (``delta_timestamps``, ``tolerance_s``) are
        taken from ``self``.  Subclasses may override this for extra per-shard
        setup (e.g. loading instruction segments).

        When ``enable_fast_init=True``, ``LeRobotDatasetMetadata`` (a pure-IO
        read of ``info.json`` + ``episodes.parquet`` + ``tasks.parquet``) is
        prefetched in a thread pool and handed to the order-sensitive
        serial register loop via ``prefetched_meta=``.  Shard count scales
        the speedup; for single-shard datasets the two paths are
        equivalent.

        Args:
            indices: Which entries of ``_all_shard_roots`` to register.
                ``None`` means all.
        """
        if indices is None:
            indices = list(range(len(self._all_shard_roots)))
        if not indices:
            return

        roots = [self._all_shard_roots[i] for i in indices]

        if self._enable_fast_init:
            # ``_ensure_hf_hub_offline`` already ran in ``__init__`` and is
            # idempotent; no need to re-invoke here.
            workers = max(1, min(self._fast_init_max_workers, len(roots)))
            metas: list[LeRobotDatasetMetadata | None] = _parallel_map(
                lambda root: LeRobotDatasetMetadata(repo_id="local", root=root, revision="local"),
                roots,
                max_workers=workers,
                label=f"{type(self).__name__}: LeRobotDatasetMetadata prefetch",
            )
        else:
            metas = [None] * len(roots)

        for root, meta in zip(roots, metas):
            label = root.rsplit("/", 1)[-1] if "/" in root else root
            self._register_source(
                root=root,
                delta_timestamps=self._delta_timestamps,
                tolerance_s=self._tolerance_s,
                dataset_label=label,
                prefetched_meta=meta,
            )

    # -- lazy dataset access -------------------------------------------------

    def _get_dataset(self, ds_idx: int) -> LeRobotDataset:
        """Get or lazily construct the LeRobot dataset for the given source index.

        Loaded datasets are tracked with LRU ordering.  When the number of
        loaded datasets exceeds ``_max_loaded_datasets`` the least-recently-used
        dataset is evicted (set back to ``None``) so the GC can reclaim it.
        """
        ds = self._datasets[ds_idx]
        if ds is not None:
            self._loaded_lru.move_to_end(ds_idx)
            return ds

        _ensure_hf_hub_offline()

        build_args = self._dataset_build_args[ds_idx]
        if build_args is None:
            raise RuntimeError(f"Missing dataset build args for dataset index {ds_idx}")

        # Evict least-recently-used datasets before loading a new one.
        while len(self._loaded_lru) >= self._max_loaded_datasets:
            evict_idx, _ = self._loaded_lru.popitem(last=False)
            self._datasets[evict_idx] = None

        with rss_tracker(
            f"[WORKER {_os.getpid()}] Lazy-loaded ds[{ds_idx}]",
            enabled=self._memprofile,
            extras_fn=lambda: [f"total loaded={len(self._loaded_lru)}/{len(self._datasets)}"],
        ):
            delta_ts = build_args["delta_timestamps"]
            if self._skip_video_loading:
                # Covers both LeRobot v2 (``observation.images.<name>``) and
                # v3 (``observation.image.<name>``) video-column conventions.
                delta_ts = {k: v for k, v in delta_ts.items() if not k.startswith("observation.image")}

            log.info(f"Loading shard root={build_args['root']}")
            ds = LeRobotDataset(
                repo_id=build_args["repo_id"],
                root=build_args["root"],
                delta_timestamps=delta_ts,
                tolerance_s=build_args["tolerance_s"],
                force_cache_sync=build_args["force_cache_sync"],
                download_videos=build_args["download_videos"],
                video_backend=build_args["video_backend"],
                revision=build_args["revision"],
                episodes=None,
            )
            if self._skip_video_loading:
                ds.meta.info["features"] = {
                    k: v for k, v in ds.meta.info["features"].items() if v.get("dtype") != "video"
                }
            self._datasets[ds_idx] = ds
            self._loaded_lru[ds_idx] = None

        return ds

    # -- index resolution ----------------------------------------------------

    def _resolve_index(self, idx: int) -> tuple[int, int, int, int]:
        """Map a flat global index to the source dataset, row, episode, and frame.

        Multiple datasets are concatenated into a single virtual sequence.
        Each episode contributes a contiguous *span* of valid frames, and
        ``_episode_cum_ends[i]`` stores the running total of valid frames
        through the *i*-th span.  For example, with two episodes of lengths
        5 and 3 the cum-ends are ``[5, 8]``, so global index 6 falls in the
        second span at offset 1.

        The lookup is O(log N) via :func:`bisect_right`.

        Returns:
            dataset_idx: Which source dataset this sample belongs to.
            row_idx: Row index *within* that dataset's LeRobot table.
            episode_id: The episode ID for this sample.
            frame_offset: Frame offset from the start of the episode span
                (0-based).

        Pure index math -- no I/O or dataset access.  Higher-level helpers
        like :meth:`_fetch_sample` build on this.
        """
        # Support negative indexing (e.g. -1 → last sample).
        if idx < 0:
            idx += self._num_valid_indices
        if idx < 0 or idx >= self._num_valid_indices:
            raise IndexError(f"{self.__class__.__name__} index {idx} out of range for size {self._num_valid_indices}")

        # _episode_cum_ends is a monotonically increasing list where entry i
        # holds the cumulative number of valid frames up to and including the
        # i-th episode span.  bisect_right finds the first span whose
        # cumulative end is strictly greater than idx, i.e. the span that
        # contains idx.
        #
        # Example: cum_ends = [5, 8, 20]
        #   idx=0  -> span_idx=0  (first span,  frames 0..4)
        #   idx=4  -> span_idx=0
        #   idx=5  -> span_idx=1  (second span, frames 5..7)
        #   idx=8  -> span_idx=2  (third span,  frames 8..19)
        span_idx = bisect_right(self._episode_cum_ends, idx)

        # The global index where this span begins is the previous span's
        # cumulative end (or 0 for the very first span).  The frame_offset
        # is how far idx is into this particular episode.
        span_start = 0 if span_idx == 0 else self._episode_cum_ends[span_idx - 1]
        frame_offset = idx - span_start

        # _episode_records[span_idx] stores (dataset_idx, row_start, valid_len,
        # episode_id).  row_start is the absolute row in the LeRobot table
        # where this episode begins.  With sample_stride=k, consecutive
        # valid indices map to rows k apart inside the episode, so the
        # effective row is row_start + frame_offset * sample_stride.
        dataset_idx, row_start, _, episode_id = self._episode_records[span_idx]
        row_idx = row_start + frame_offset * self._sample_stride
        return dataset_idx, row_idx, episode_id, frame_offset

    def _choose_mode(self) -> str:
        """Resolve the active mode for one sample request."""
        if self._mode == "joint":
            return random.choice(("forward_dynamics", "inverse_dynamics", "wam"))
        return self._mode

    def _fetch_sample(self, idx: int) -> tuple[str, int, int, dict[str, Any]]:
        """Resolve index, pick a mode, and load the sample from the dataset.

        Returns ``(mode, dataset_idx, row_idx, sample_dict)``.
        """
        mode = self._choose_mode()
        dataset_idx, row_idx, _, _ = self._resolve_index(idx)

        self._getitem_count = getattr(self, "_getitem_count", 0) + 1
        profile = self._memprofile and self._getitem_count % 50 == 1

        with rss_tracker(
            f"[WORKER {_os.getpid()}] __getitem__ transient (dataset_idx={dataset_idx})",
            enabled=profile,
            after_fn=lambda: log_worker_memory_breakdown(self),
        ):
            sample = self._get_dataset(dataset_idx)[row_idx]

        if self._skip_video_loading:
            sample = defaultdict(lambda: None, sample)

        return mode, dataset_idx, row_idx, sample

    # -- action normalization ------------------------------------------------

    def _normalizer_filename(self) -> str:
        """Bundled stats filename for this dataset instance.

        Default convention (matches ``compute_action_stats.py`` output):
        ``<embodiment_type>[_<pose_convention>][_<rotation_format>].json``.

        Pose/rotation suffixes are appended only when the instance actually
        has them (SE(3) pose datasets like Bridge / DROID).  Joint-space
        datasets — where both are ``None`` — resolve to just
        ``<embodiment_type>.json``.

        Subclasses may override when the bundled filename uses a different
        scheme (e.g. UMI's ``uva_umi_single_task_normalizer.json``).
        """
        if not self._embodiment_type:
            raise RuntimeError(
                f"{self.__class__.__name__}: embodiment_type is not set; cannot resolve normalizer filename."
            )
        parts = [self._embodiment_type]
        if self._pose_convention:
            parts.append(self._pose_convention)
        if self._rotation_format:
            parts.append(self._rotation_format)
        return "_".join(parts) + ".json"

    def _normalizer_path(self) -> Path:
        """Full path to the bundled stats JSON for this dataset."""
        return self._NORMALIZERS_DIR / self._normalizer_filename()

    def _load_norm_stats(self, action_normalization: ActionNormalization) -> dict[str, torch.Tensor]:
        """Load action normalization stats for the configured normalization mode.

        Raises :class:`FileNotFoundError` if the stats file is missing.  This
        is intentional — silently falling back to identity normalization when
        the user asked for ``quantile`` / ``quantile_rot`` / ``meanstd`` /
        ``minmax`` would be a training bug.
        """
        stats_key = "global_raw" if action_normalization == "quantile_rot" else "global"
        raw_stats = load_action_stats(str(self._normalizer_path()), stats_key=stats_key)
        return {key: torch.from_numpy(value).float() for key, value in raw_stats.items()}  # dict[str,[D]]

    def get_action_normalizer(
        self,
        _sample: dict[str, Any] | None = None,
    ) -> ActionNormalizer | None:
        """Return the configured action normalizer for transform-time preprocessing."""
        return self._action_normalizer

    # -- video formatting ----------------------------------------------------

    def _convert_video(self, video_tchw: torch.Tensor | None) -> torch.Tensor | None:
        """Convert LeRobot ``(T,C,H,W)`` float video to Action ``(C,T,H,W)`` uint8.

        Args:
            video_tchw: Raw floating-point video tensor in ``[0, 1]`` with
                LeRobot layout, or ``None``.  # [T,C,H,W] | None

        Returns:
            Action-formatted video tensor, or ``None``.  # [C,T,H,W] | None
        """
        if self._skip_video_loading or video_tchw is None:
            return None
        if video_tchw.ndim != 4:
            raise ValueError(
                f"{self.__class__.__name__}._convert_video expected video with shape [T,C,H,W], "
                f"got ndim={video_tchw.ndim}"
            )
        if not torch.is_floating_point(video_tchw):
            raise TypeError(
                f"{self.__class__.__name__}._convert_video expected floating-point video in [0, 1], "
                f"got dtype={video_tchw.dtype}"
            )
        video_min = video_tchw.amin()  # []
        video_max = video_tchw.amax()  # []
        if video_min.item() < 0.0 or video_max.item() > 1.0:
            raise ValueError(
                f"{self.__class__.__name__}._convert_video expected floating-point video in [0, 1], "
                f"got range=[{video_min.item():.6f}, {video_max.item():.6f}]"
            )
        formatted_video = (video_tchw * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)  # [C,T,H,W]
        return formatted_video

    # -- result building -----------------------------------------------------

    def _build_action_spec(self) -> ActionSpec | None:
        """Subclass override: declare this dataset's action layout.

        Called once per instance — the result is cached by ``self.action_spec``.
        Return ``None`` to skip spec-driven idle detection; in that case
        ``_compute_idle_frames`` will log a one-time warning and return
        ``None`` for every sample.
        """
        return None

    @cached_property
    def action_spec(self) -> ActionSpec | None:
        """Cached :class:`ActionSpec` from ``_build_action_spec``.

        Returns ``None`` when the subclass did not declare one; idle detection
        is then skipped (with a one-time warning) until the subclass overrides
        ``_build_action_spec``.
        """
        return self._build_action_spec()

    @cached_property
    def action_names(self) -> list[str] | None:
        spec = self.action_spec
        return spec.names if spec is not None else None

    # Idle-detection thresholds. Defined as **velocities** (per second) so the
    # same numeric value means the same physical motion across datasets with
    # different sampling rates; converted to per-frame at call time using
    # ``self._fps`` via :meth:`_resolve_idle_thresholds`.
    #
    # Defaults:
    #   - ``idle_eps_t_per_sec``           = 5 mm/s   (≈ 1 mm/frame at 5 Hz)
    #   - ``idle_eps_r_per_sec``           = 1.5°/s   (geodesic, rotation-format aware)
    #   - ``idle_eps_g``                   = 1e-2     unit gripper Δ (no fps)
    #   - ``idle_joint_threshold_per_sec`` = 5e-3 rad/s
    #   - ``idle_min_streak``              = 3        require ≥ 3 consecutive
    #
    # Subclasses can either override the ``*_per_sec`` attributes (preferred —
    # keeps the velocity semantics) or set the corresponding ``idle_eps_*`` /
    # ``idle_joint_threshold`` attribute to a non-``None`` value to bypass the
    # per-fps conversion entirely (raw per-frame override).
    idle_eps_t_per_sec: float = 5e-3
    idle_eps_r_per_sec: float = math.radians(1.5)
    idle_eps_g: float = 1e-2
    idle_joint_threshold_per_sec: float = 5e-3
    idle_min_streak: int = 3

    # Optional per-frame overrides. ``None`` (default) → use the ``*_per_sec``
    # attribute / fps conversion above.
    idle_eps_t: float | None = None
    idle_eps_r: float | None = None
    idle_joint_threshold: float | None = None

    def _resolve_idle_thresholds(self) -> tuple[float, float, float, float]:
        """Resolve per-frame idle thresholds for this dataset instance.

        Returns ``(eps_t, eps_r, eps_g, joint_threshold)`` in raw per-frame
        units. Honours direct per-frame overrides if the subclass sets the
        non-``_per_sec`` attribute; otherwise scales the ``_per_sec`` values
        by ``self._fps``.
        """
        fps = float(self._fps) if self._fps else 1.0
        eps_t = self.idle_eps_t if self.idle_eps_t is not None else self.idle_eps_t_per_sec / fps
        eps_r = self.idle_eps_r if self.idle_eps_r is not None else self.idle_eps_r_per_sec / fps
        joint_thr = (
            self.idle_joint_threshold
            if self.idle_joint_threshold is not None
            else self.idle_joint_threshold_per_sec / fps
        )
        return float(eps_t), float(eps_r), float(self.idle_eps_g), float(joint_thr)

    def _compute_idle_frames(self, raw_action: torch.Tensor) -> torch.Tensor | None:
        """Count idle frames in the *raw* (un-normalized) action chunk.

        Requires ``self.action_spec`` to be declared via ``_build_action_spec``.
        Returns ``None`` when:
        - ``pose_convention`` is not ``"backward_framewise"`` (TODO: extend),
        - the subclass has not declared an ``ActionSpec`` (logs a one-time warning),
        - the action layout does not match the declared spec.

        Detection thresholds come from the ``idle_eps_*`` class attributes
        (overridable per dataset). Subclasses can also override this method
        outright, or pass an explicit ``idle_frames`` integer via
        ``**extras`` to :meth:`_build_result`.
        """
        # TODO: currently we only support backward_framewise. Other pose
        # conventions (anchored / absolute) need different idle semantics.
        if self._pose_convention != "backward_framewise":
            if not getattr(self, "_warned_pose_convention", False):
                log.warning(
                    f"Dataset {self.__class__.__name__}: pose_convention="
                    f"{self._pose_convention!r} is not 'backward_framewise'; "
                    "skipping idle-frames detection. Centralize the dataset "
                    "to backward_framewise to enable IdleFrames captioning."
                )
                self._warned_pose_convention = True
            return None

        spec = self.action_spec
        if spec is None:
            if not getattr(self, "_warned_no_action_spec", False):
                log.warning(
                    f"Dataset {self.__class__.__name__} has no action spec defined; "
                    "skipping idle-frames detection. Override _build_action_spec() to enable it."
                )
                self._warned_no_action_spec = True
            return None

        eps_t, eps_r, eps_g, joint_thr = self._resolve_idle_thresholds()
        try:
            n = compute_idle_frames(
                raw_action,
                spec,
                eps_t=eps_t,
                eps_r=eps_r,
                eps_g=eps_g,
                joint_threshold=joint_thr,
                min_streak=self.idle_min_streak,
            )
        except (ValueError, TypeError) as e:
            if not getattr(self, "_warned_action_layout", False):
                log.warning(
                    f"Dataset {self.__class__.__name__}: action layout does "
                    f"not match the declared ActionSpec "
                    f"(action_dim={int(raw_action.shape[-1])}, "
                    f"spec.dim={spec.dim}); skipping idle-frames detection. "
                    f"Underlying error: {e}"
                )
                self._warned_action_layout = True
            return None
        return torch.tensor(n, dtype=torch.long)

    def _build_result(
        self,
        *,
        mode: str,
        video: torch.Tensor | None,
        action: torch.Tensor,
        ai_caption: str,
        **extras: Any,
    ) -> dict[str, Any]:
        """Assemble the common return dict for ``__getitem__``.

        ``video`` is expected in raw LeRobot layout before final formatting.
        Subclasses may pass extra keys (e.g. ``initial_pose``) via ``**extras``.
        ``idle_frames`` is auto-computed from the raw (un-normalized) ``action``
        whenever the dataset's pose/rotation conventions allow it; subclasses
        can override by passing ``idle_frames`` (int or scalar tensor) via
        ``**extras``.
        """
        # Compute idle_frames from the raw action before normalization, unless
        # the subclass has provided one explicitly via ``**extras``.
        if "idle_frames" not in extras:
            idle_frames = self._compute_idle_frames(action)
            if idle_frames is not None:
                extras = {"idle_frames": idle_frames, **extras}

        if self._action_normalizer is not None:
            action = self._action_normalizer.normalize_action(action)  # [T,D]

        if self._skip_video_loading:
            result: dict[str, Any] = {"action": action}
            if "idle_frames" in extras:
                result["idle_frames"] = extras["idle_frames"]
            return result
        formatted_video = self._convert_video(video)  # [C,T,H,W] | None
        return {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "viewpoint": self._viewpoint,
            **extras,
        }

    def __len__(self) -> int:
        return self._num_valid_indices

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode (or per kept-segment when ``use_filter_dict``) flat-index
        blocks ``(start, length)`` over ``[0, len(self))``. ``ActionIterableShuffleDataset``
        shuffles the ORDER of these blocks and shards them disjointly across
        ``(rank, worker)`` while keeping windows *within* a block sequential ->
        decorrelates batches without random-access I/O. Derived from
        ``_episode_cum_ends`` (the same monotonic cumulative index ``__getitem__``
        bisects), so blocks align exactly with flat-index addressing."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in self._episode_cum_ends:
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

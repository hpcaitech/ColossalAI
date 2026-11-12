"""Map-style DROID action-policy dataset for the Cosmos 3 SFT recipe.

Follows the preprocessing of ``action_policy_droid_nano`` in nvidia/cosmos-framework:
``concat_view`` (wrist on top, the two shoulder views side by side at half width below), resized onto the closest
predefined canvas, then reflection-padded at the bottom and right. ``chunk_length=32`` gives 33 observation frames
and 32 actions, at 15 fps, in the unnormalized 8-dimensional ``joint_pos`` action space.

Three design decisions worth stating, because each is easy to get wrong in the other direction:

1. **Samples carry pixels, not latents.** The VAE runs inside the training loop, as it does upstream, because a
   DataLoader worker process cannot touch an NPU device. Image augmentation also has to happen before the VAE, so
   encoding ahead of time would freeze the augmentation into the cached latents.

2. **Index order is episode-shuffled and sequential within an episode**, to be used with a ``shuffle=False``
   sampler. Upstream explains why: purely sequential sampling hands one rank a run of overlapping windows, so the
   global batch covers about one episode and the gradient norm becomes unstable, while a random sampler decorrelates
   the batch at the cost of random-access I/O. Shuffling at index-construction time and leaving the sampler
   sequential gives both the decorrelation and the sequential reads.

3. **The released conversion has two roots and a curated window set.** ``LEROBOT_ROOTS`` for this layout is
   ``["success", "failure"]``. The base ``DROIDLeRobotDataset`` defaults ``use_success_only=False`` while the
   ``action_policy_droid_nano`` factory defaults it to True, and the Cosmos 3 technical report describes
   failure-demonstration removal as part of the policy recipe. The registered config leaves the curated filter
   off (``use_filter_dict=False``, ``filter_dict_path=None``), but the official launch script
   (``launch_sft_action_policy_droid_nano.sh``) overrides both to load ``keep_ranges_1_0_1.json``, and the report
   describes community-provided idle-frame filtering. Reproducing the reference run therefore means the success
   root with the curated windows; this loader defaults to both roots and no filter so a smoke run needs nothing
   else, and exposes ``success_only``/``keep_ranges`` for the recipe settings.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

# cosmos-framework VIDEO_RES_SIZE_INFO: values are (width, height) and every side is a multiple of 16.
RES_TABLE = {
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
        "2,3": (512, 768),
    },
}
CAMS = ("wrist", "left", "right")
CAM_FEATURE = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}
JOINT_ACTION = "action.joint_position"
GRIPPER_ACTION = "action.gripper_position"
# The state has to live in the action space: the recipe sets action_space="joint_pos", so an action is
# joint(7) + gripper(1). The state is prepended to the action stream as the conditioning "initial state" step
# (use_state=True), which rules out cartesian_position: it is 6-dimensional and in a different space.
STATE_JOINT = "observation.state.joint_positions"
STATE_GRIPPER = "observation.state.gripper_position"

# The curated window set predates the LeRobot conversion and is keyed by the raw DROID trajectory path, which the
# conversion kept as the episodes table's ``episode_id``. The original files use a composite URI key; the official
# loader normalizes such keys back to the relative episode id when an explicit source prefix is supplied.
_KEEP_RANGES_KEY = (
    "gs://xembodiment_data/r2d2/r2d2-data-full/{episode}/recordings/"
    "MP4--gs://xembodiment_data/r2d2/r2d2-data-full/{episode}/trajectory.h5"
)
DROID_FILTER_SOURCE_PREFIX = "DROID_FILTER_SOURCE_PREFIX"

logger = logging.getLogger(__name__)


def split_episode_ids(total_episodes: int, seed: int = 42, val_ratio: float = 0.03, split: str = "train") -> list[int]:
    """Episode-level train/val ids, the same permutation cosmos-framework's DROID loader draws.

    Upstream keeps this in ``cosmos_framework/data/generator/action/datasets/cosmos3_action_lerobot.py`` and
    every DROID recipe goes through it, so the 3% held out here is exactly the 3% held out there. That
    matters beyond tidiness: a held-out number is only comparable across the two codebases if it was measured
    on the same episodes, and the recipe itself never evaluates its val split (``dataloader_val=None``) — the
    split exists so the validation episodes stay out of training, and the ids are what define it.

    Upstream draws this once per root, not once per dataset: success and failure are separate LeRobot roots with
    their own episode tables, so each gets its own permutation from the same seed.

    Args:
        total_episodes (int): Number of episodes in the dataset, i.e. ``meta/info.json``'s ``total_episodes``.
        seed (int, optional): Permutation seed. Defaults to 42, the recipe's value.
        val_ratio (float, optional): Fraction held out. Defaults to 0.03, the recipe's value.
        split (str, optional): ``"train"``, ``"val"`` or ``"full"``. ``"full"`` returns the whole permutation
            in shuffled order, which is what an evaluation over every episode wants.

    Returns:
        list[int]: Episode indices (positions in the episodes table, which LeRobot numbers 0..N-1).

    Raises:
        ValueError: If ``split`` is not recognised.
    """
    num_val = int(round(total_episodes * val_ratio))
    episode_ids = torch.randperm(total_episodes, generator=torch.Generator().manual_seed(seed)).tolist()
    if split == "train":
        return episode_ids[num_val:]
    if split == "val":
        return episode_ids[:num_val]
    if split == "full":
        return episode_ids
    raise ValueError(f"Unsupported split {split!r}; expected 'train', 'val' or 'full'.")


def resolve_shards(root: str | Path, success_only: bool = False) -> list[tuple[str, Path]]:
    """The LeRobot roots one training run reads from.

    Point ``root`` at the parent directory and both roots are indexed, the base ``DROIDLeRobotDataset`` default;
    point it at ``success`` or ``failure`` itself and only that one is. The SFT factory restricts to ``success``
    by default (``use_success_only=True``). The order is upstream's ``LEROBOT_ROOTS`` order, for the same reason
    it matters there: it decides which root an episode index refers to when the two are mixed.

    Args:
        root (str | Path): The parent of ``success/`` and ``failure/``, or one of the two itself.
        success_only (bool, optional): Drop the failure root, i.e. upstream's ``use_success_only``. Defaults to
            False, the base dataset's value; the recipe factory passes True.

    Returns:
        list[tuple[str, Path]]: ``(name, path)`` per root, names unique.

    Raises:
        FileNotFoundError: If ``root`` is neither a LeRobot root nor a parent of one or more.
    """
    root = Path(root)
    shards = [(name, root / name) for name in ("success", "failure") if (root / name / "meta/info.json").is_file()]
    if shards:
        if success_only:
            shards = [shard for shard in shards if shard[0] == "success"]
        if not shards:
            raise FileNotFoundError(f"{root} holds no success root, and success_only was requested.")
        return shards
    if (root / "meta/info.json").is_file():
        if success_only and root.name != "success":
            raise FileNotFoundError(f"{root} is not a success root, and success_only was requested.")
        return [(root.name, root)]
    raise FileNotFoundError(f"{root} is neither a LeRobot root nor a parent of success/ and failure/.")


def keep_ranges_key(episode_id: str) -> str:
    """The legacy composite URI key of one episode in ``keep_ranges_1_0_1.json``.

    Older files list episodes under this key. It is preserved so such files keep working, but it is not the
    identity the loader queries with: every key is normalized back to the relative episode id first.

    Args:
        episode_id (str): The episodes table's ``episode_id``, e.g. ``"AUTOLab/success/2023-07-07/Fri_Jul__7_..."``.

    Returns:
        str: The legacy URI key the curated window set uses for that episode.
    """
    return _KEEP_RANGES_KEY.format(episode=episode_id)


def normalize_episode_filter_keys(filters: dict[str, Any], *, source_prefix: str = "") -> dict[str, Any]:
    """Reduce every filter key to the relative episode id, the protocol the official loader uses.

    Port of ``normalize_episode_filter_keys`` in cosmos-framework's
    ``data/generator/action/utils/episode_filters.py`` at ``5950e7b4``. A complete relative episode id is used as
    it is. A composite ``gs://`` URI key is reduced by stripping the ``/recordings/MP4--...`` suffix and then an
    explicit ``source_prefix``; without a prefix a URI key is rejected rather than guessed, because a wrong guess
    would silently select the wrong episodes. Identical duplicate entries collapse; conflicting ranges for one
    episode raise.

    Args:
        filters (dict[str, Any]): The JSON content of a keep-ranges file, ``{key: [[start, end], ...]}``.
        source_prefix (str, optional): Dataset URI prefix in front of the complete episode id, e.g.
            ``"gs://xembodiment_data/r2d2/r2d2-data-full"``. Required only for URI keys.

    Returns:
        dict[str, Any]: ``{episode_id: ranges}``.

    Raises:
        ValueError: If a URI key has no prefix or the prefix does not match it, if an entry identifies no
            episode, or if two entries give one episode different ranges.
    """
    normalized: dict[str, Any] = {}
    for key, ranges in filters.items():
        episode = key
        if "://" in key:
            # Relative episode ids are opaque, including any "--" in their names; only composite URI keys carry
            # the recording/trajectory delimiter.
            recording = key.split("/recordings/MP4--", 1)[0].rstrip("/")
            if not source_prefix:
                raise ValueError(
                    "filter_dict_source_prefix is required for URI-based episode filter keys. Set "
                    f"{DROID_FILTER_SOURCE_PREFIX} when using a DROID launcher, or pass "
                    "filter_dict_source_prefix to the dataset. Use the URI prefix before the complete episode id."
                )
            prefix = source_prefix.rstrip("/") + "/"
            if not recording.startswith(prefix):
                raise ValueError(
                    "Episode filter URI does not match filter_dict_source_prefix. Check "
                    f"{DROID_FILTER_SOURCE_PREFIX} against the keys in your filter JSON."
                )
            episode = recording.removeprefix(prefix)
        if not episode:
            raise ValueError("Filter entries must identify an episode")
        if episode in normalized and normalized[episode] != ranges:
            raise ValueError(f"Conflicting filter ranges for episode {episode!r}")
        normalized[episode] = ranges
    return normalized


def episode_starts(length: int, chunk_length: int, stride: int = 1) -> list[int]:
    """Window starts, relative to the episode, when nothing filters them.

    ``length - chunk_length`` starts fit, the same count upstream's ``build_episode_spans`` computes
    (``dataset_to_index - dataset_from_index - chunk_length``).

    Args:
        length (int): Frames in the episode, the episodes table's ``length``.
        chunk_length (int): Actions per window; a window also reads one frame more than that.
        stride (int, optional): Step between window starts. Defaults to 1.

    Returns:
        list[int]: Offsets of the first frame of each window.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    usable = length - chunk_length
    if usable <= 0:
        return []
    return [index * stride for index in range((usable + stride - 1) // stride)]


def filtered_starts(ranges: list[list[int]], length: int, chunk_length: int, stride: int = 1) -> list[int]:
    """Window starts inside the curated ``[start, end)`` frame ranges of one episode.

    Transcribed from upstream's ``_append_index_records``: a range keeps the starts in
    ``[start, end - chunk_length)``, clamped to the episode. Ranges are not merged there, so overlapping ones are
    not merged here either — merging would quietly train a different set of windows than the released run. With
    ``stride=1`` these are exactly upstream's starts. ``stride>1`` is an extension of this loader: all bounds are
    frame offsets, so the step is anchored at the clamped range origin and the upper bound is not converted a
    second time; it is not a transcription of the upstream filter loop.

    Args:
        ranges (list[list[int]]): ``[start, end]`` pairs from the curated window set.
        length (int): Frames in the episode.
        chunk_length (int): Actions per window.
        stride (int, optional): Step between window starts. Defaults to 1.

    Returns:
        list[int]: Offsets of the first frame of each kept window.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    starts = []
    for start, end in ranges:
        # All bounds are frame offsets. Stride is the distance between starts, anchored to the
        # clamped beginning of each range, not a second conversion of the upper bound.
        sub_start = max(int(start), 0)
        sub_end = min(int(end), length) - chunk_length
        starts.extend(range(sub_start, sub_end, stride))
    return starts


# Verbatim from droid_lerobot_dataset.py, where the dataset appends it to the generic ``concat_view`` template
# through ``extras["additional_view_description"]`` for ``viewpoint == "concat_view"``. The generic template only
# states that the views are concatenated; these two sentences are what tell the model which camera is in which
# row, and they are 33 of the 45 prompt tokens we were missing.
CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person perspective views of the scene from "
    "opposite sides, with the robot visible."
)

# Idle-frame thresholds, resolved the way ``Cosmos3ActionLeRobotDataset._resolve_idle_thresholds`` resolves them:
# the per-second values are divided by fps. These are deliberately not ``compute_idle_frames``' own signature
# defaults -- that function declares ``joint_threshold=5e-4``, but the LeRobot caller always passes a resolved
# value, so at 15 fps the real threshold is 5e-3 / 15 = 3.33e-04 rad/step. DROID overrides none of them.
IDLE_JOINT_THRESHOLD_PER_SEC = 5e-3
IDLE_EPS_GRIPPER = 1e-2  # a gripper command delta rather than a rate, so upstream does not scale it by fps
IDLE_MIN_STREAK = 3


def _keep_streaks(idle: list[bool], min_streak: int) -> list[bool]:
    """Clear idle flags that are not part of a run of at least ``min_streak`` consecutive ones.

    Upstream's ``pose_utils._consecutive_streaks``. The filter is there because an isolated low-motion frame is an
    instantaneous slowdown rather than the robot waiting, and those carry little physical meaning.

    Args:
        idle (list[bool]): Per-frame idle flags.
        min_streak (int): Minimum run length to keep. ``<= 1`` is a no-op.

    Returns:
        list[bool]: The filtered flags.
    """
    if min_streak <= 1:
        return idle
    out = [False] * len(idle)
    i = 0
    while i < len(idle):
        if not idle[i]:
            i += 1
            continue
        j = i
        while j < len(idle) and idle[j]:
            j += 1
        if j - i >= min_streak:
            out[i:j] = [True] * (j - i)
        i = j
    return out


def count_idle_frames(action: torch.Tensor, fps: float) -> int:
    """Count idle frames in an unnormalized ``[T, 8]`` DROID ``joint_pos`` action chunk.

    This is ``pose_utils.compute_idle_frames`` specialised to the action spec DROID declares for ``joint_pos``,
    ``build_action_spec(Joint(n=7), Gripper())`` -- seven JOINT dims and one GRIPPER dim, no POS and no ROT, so the
    general function's translation and rotation branches have nothing to act on here. A frame is idle when **both**
    groups are idle on it, and it only counts if it belongs to a run of at least :data:`IDLE_MIN_STREAK` such
    frames.

    Both groups use upstream's frame-difference scheme: a diff with the first row prepended, so step 0 compares
    against itself and is always "no change". The action must be unnormalized, because the thresholds are in
    radians per step and in gripper command units.

    Args:
        action (torch.Tensor): ``[T, 8]`` unnormalized action -- the same chunk the prompt describes, which means
            with the initial state row already prepended, since upstream computes this after that prepend.
        fps (float): Frame rate, which turns the per-second thresholds into per-step ones.

    Returns:
        int: The idle frame count, for the prompt's ``idle_frame`` field.
    """
    if action.ndim != 2 or action.shape[-1] != 8:
        raise ValueError(f"expected an unnormalized [T, 8] joint_pos action chunk, got {tuple(action.shape)}")
    x = action.detach().float()
    joint_thr = IDLE_JOINT_THRESHOLD_PER_SEC / float(fps)
    joints, grip = x[:, :7], x[:, 7:]
    j_idle = torch.diff(joints, dim=0, prepend=joints[:1]).abs().amax(dim=1) < joint_thr
    g_idle = torch.diff(grip, dim=0, prepend=grip[:1]).abs().amax(dim=1) < IDLE_EPS_GRIPPER
    return int(sum(_keep_streaks((j_idle & g_idle).tolist(), IDLE_MIN_STREAK)))


@dataclass(frozen=True)
class Window:
    """One training window: 33 observation frames and 32 actions starting at ``start`` inside an episode.

    ``root`` is the physical LeRobot root the episode lives in — success and failure are separate roots — and
    ``root_role`` is its logical role from :func:`resolve_shards` (``"success"``/``"failure"`` in the released
    layout). ``episode_id`` is the raw DROID trajectory id, the relative id the curated window set is keyed by.
    ``draw_id`` counts repeated draws of one logical window when overlapping curated ranges list it twice.
    """

    episode_index: int
    episode_id: str
    root: Path
    root_role: str
    start: int  # first frame, relative to the episode
    task: str
    data_chunk: int
    data_file: int
    row_from: int  # first row of this episode **within its own parquet file**
    video: dict[str, tuple[int, int, float]]  # cam -> (chunk, file, from_timestamp)
    draw_id: int = 0  # occurrence index when overlapping curated ranges repeat one window


def closest_canvas(h: int, w: int, resolution: str = "480") -> tuple[int, int, str]:
    """Pick the predefined canvas with the closest aspect ratio; returns ``(target_h, target_w, aspect_key)``."""
    table = RES_TABLE[str(resolution)]
    ratio = h / w
    key, (tw, th) = min(table.items(), key=lambda kv: abs(ratio - kv[1][1] / kv[1][0]))
    return th, tw, key


def resize_and_reflect_pad(x: torch.Tensor, target_h: int, target_w: int):
    """Scale to fit the canvas, then reflection-pad bottom and right to the exact size.

    Returns:
        tuple: The padded tensor and the ``(h, w)`` of the content region inside it.
    """
    h, w = x.shape[-2:]
    # The 1.0 cap is upstreams, and it is not cosmetic: reflection_pad_to_target computes
    # min(target_w / orig_w, target_h / orig_h, 1.0), so content smaller than the canvas is padded, never
    # stretched. DROIDs concat_view composite is 540x640 against a 544x736 canvas, where the uncapped ratio is
    # 1.0074 -- enough to resample every pixel for the sake of four rows.
    scale = min(target_h / h, target_w / w, 1.0)
    rh, rw = int(scale * h + 0.5), int(scale * w + 0.5)
    if (rh, rw) != (h, w):
        # Upstream resizes with torchvisions BICUBIC + antialias=True. With the cap above this path no longer
        # runs for DROID, but the mode still has to match for any input that does need downscaling.
        x = F.interpolate(x, size=(rh, rw), mode="bicubic", align_corners=False, antialias=True)
    pad_b, pad_r = target_h - rh, target_w - rw
    if pad_b or pad_r:
        # Reflection padding requires the pad to be smaller than the side it mirrors; fall back to edge replication.
        mode = "reflect" if pad_b < rh and pad_r < rw else "replicate"
        x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
    return x, (rh, rw)


def compose_concat_view(wrist: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Upstream geometry: the wrist view keeps its size on top, the two shoulder views go below at half width.

    Each shoulder view is scaled to half the wrist width with its own aspect ratio preserved, so the height of the
    bottom strip follows from the source views rather than being fixed.

    Returns:
        torch.Tensor: ``[T, C, H + H_bottom, W]``.
    """
    _, _, h, w = wrist.shape
    half_w = w // 2
    out = []
    for view in (left, right):
        vh, vw = view.shape[-2:]
        th = round(vh * half_w / vw)
        out.append(F.interpolate(view, size=(th, half_w), mode="bilinear", align_corners=False))
    bottom = torch.cat(out, dim=-1)
    return torch.cat([wrist, bottom], dim=-2)


class DroidActionPolicyDataset(Dataset):
    """DROID windows in LeRobot v3 layout, preprocessed the way the action-policy recipe expects.

    Args:
        root (str | Path): The parent directory holding ``success/`` and ``failure/`` — both are trained — or one
            of those two on its own. See :func:`resolve_shards`.
        chunk_length (int, optional): Number of actions per window; the window holds ``chunk_length + 1``
            observation frames. Defaults to 32.
        fps (float, optional): Frame rate the video timestamps are resolved against. Defaults to 15.0.
        resolution (str, optional): Key into :data:`RES_TABLE`. Defaults to "480".
        sample_stride (int, optional): Step between window starts inside an episode. Defaults to 1, upstream's
            value; values above one are an extension of this loader.
        episode_shuffle_seed (int, optional): Seed of the episode-order shuffle. Defaults to 42.
        max_episodes (int | None, optional): Index only the first N episodes after the shuffle and after the
            curated filter, for smoke runs. Defaults to None.
        augment (bool, optional): Apply the image augmentation. Defaults to False.
        split (str, optional): ``"train"`` (default), ``"val"`` or ``"full"``. Episode-level, ``split_seed``,
            3% held out, drawn once per root — see :func:`split_episode_ids`. The recipe never evaluates
            ``"val"``; it is what training must exclude.
        split_val_ratio (float, optional): Fraction held out. Defaults to 0.03.

    The first ten arguments keep the positional order they had before the multi-root extension; everything below
    is keyword-only.

    Keyword Args:
        split_seed (int | None, optional): Permutation seed of the split. None (default) uses
            ``episode_shuffle_seed``, matching upstream, where one ``randperm`` is both the holdout and the
            episode shuffle; pass an explicit value to give the split its own seed.
        success_only (bool, optional): Train on the success root alone, i.e. upstream's ``use_success_only``.
            Defaults to False, the base dataset's value; the ``action_policy_droid_nano`` factory sets it True,
            which is why the reference recipe excludes failure demonstrations.
        keep_ranges (str | Path | None, optional): A curated keep-ranges JSON such as
            ``keep_ranges_1_0_1.json``. The registered config leaves the filter off, but the official launch
            script enables it for the reference reproduction; the loader keeps it opt-in. When given, only
            windows inside the listed frame ranges are indexed, and episodes the file does not mention are
            dropped. Keys are normalized to relative episode ids first — see
            :func:`normalize_episode_filter_keys`. Defaults to None, which indexes every window.
        filter_dict_source_prefix (str | None, optional): URI prefix in front of the complete episode id, needed
            when a filter file uses legacy composite URI keys. When None, ``DROID_FILTER_SOURCE_PREFIX`` is read.
        dataset_revision (str, optional): Release tag placed in front of every sample id, so two releases that
            reuse episode ids do not share identities. Defaults to an empty string.
    """

    def __init__(
        self,
        root: str | Path,
        chunk_length: int = 32,
        fps: float = 15.0,
        resolution: str = "480",
        sample_stride: int = 1,
        episode_shuffle_seed: int = 42,
        max_episodes: int | None = None,
        augment: bool = False,
        split: str = "train",
        split_val_ratio: float = 0.03,
        *,
        split_seed: int | None = None,
        success_only: bool = False,
        keep_ranges: str | Path | None = None,
        filter_dict_source_prefix: str | None = None,
        dataset_revision: str = "",
    ):
        self.root = Path(root)
        self.chunk_length = chunk_length
        self.n_obs = chunk_length + 1
        self.fps = fps
        self.resolution = resolution
        self.augment = augment
        self._augmentor: T.Compose | None = None
        self.split = split
        # Upstream derives the holdout from the same randperm as the episode shuffle. None keeps that coupling
        # for callers that only pass episode_shuffle_seed; an explicit split_seed decouples the two.
        self.split_seed = episode_shuffle_seed if split_seed is None else split_seed
        self.dataset_revision = dataset_revision
        self.filter_dict_source_prefix = (
            filter_dict_source_prefix
            if filter_dict_source_prefix is not None
            else os.environ.get(DROID_FILTER_SOURCE_PREFIX, "")
        )
        self.filter_stats: dict[str, int] | None = None
        self.shards = resolve_shards(self.root, success_only)
        self.split_ids = {
            name: split_episode_ids(
                int(json.loads((path / "meta/info.json").read_text())["total_episodes"]),
                self.split_seed,
                split_val_ratio,
                split,
            )
            for name, path in self.shards
        }
        self.windows = self._build_index(sample_stride, episode_shuffle_seed, max_episodes, keep_ranges)

    @property
    def num_episodes(self) -> int:
        """Episodes in the split, summed over the roots."""
        return sum(len(ids) for ids in self.split_ids.values())

    # ---------- index ----------

    def _read_episodes(self, root: Path, root_role: str) -> list[dict]:
        """Every row of one root's episodes table, with the fields the index needs."""
        cols = ["episode_index", "episode_id", "tasks", "length", "data/chunk_index", "data/file_index"]
        cols.append("dataset_from_index")
        for cam in CAMS:
            f = CAM_FEATURE[cam]
            cols += [f"videos/{f}/chunk_index", f"videos/{f}/file_index", f"videos/{f}/from_timestamp"]
        rows: list[dict] = []
        for p in sorted((root / "meta/episodes").rglob("*.parquet")):
            rows += pq.read_table(p, columns=cols).to_pylist()

        # dataset_from_index is a global row number while each parquet file holds only one slice of the dataset.
        # Take the smallest global row number per (chunk, file) as that file's base and convert to a file-local
        # offset. Getting this wrong reads actions from the wrong episode without raising, which is why
        # _read_actions re-checks episode_index against the row it lands on.
        base: dict[tuple[int, int], int] = {}
        for r in rows:
            k = (r["data/chunk_index"], r["data/file_index"])
            v = r["dataset_from_index"]
            if k not in base or v < base[k]:
                base[k] = v
        for r in rows:
            r["_row_from"] = r["dataset_from_index"] - base[(r["data/chunk_index"], r["data/file_index"])]
            r["_root"] = root
            r["_root_role"] = root_role
        return rows

    def _build_index(self, stride, seed, max_episodes, keep_ranges) -> list[Window]:
        # Loaded only while the index is built and then dropped: the file is ~22 MB of JSON, and every spawned
        # DataLoader worker would otherwise carry its own copy for no reason.
        filter_dict = None
        if keep_ranges is not None:
            with open(keep_ranges) as f:
                filter_dict = normalize_episode_filter_keys(json.load(f), source_prefix=self.filter_dict_source_prefix)

        # Drop the episodes this split excludes before shuffling, so max_episodes truncates inside the split
        # rather than across both. LeRobot numbers episode_index 0..total_episodes-1 in table order, which is
        # what split_episode_ids permutes; filtering on episode_index therefore selects the same episodes as
        # upstream's positional indexing.
        rows: list[dict] = []
        for name, path in self.shards:
            # The physical path is logged, never embedded in a sample identity: the same release mounted at
            # another path must produce the same ids.
            logger.info("Indexing LeRobot root %r at %s", name, path.resolve())
            allowed = set(self.split_ids[name])
            by_id = {int(r["episode_index"]): r for r in self._read_episodes(path, name)}
            # Preserve upstream's torch permutation within each root. Split membership has its own seed;
            # changing the traversal order must never move a validation episode into training.
            total = int(json.loads((path / "meta/info.json").read_text())["total_episodes"])
            order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
            rows.extend(by_id[i] for i in order if i in allowed and i in by_id)
        if filter_dict is not None:
            # Dropping unlisted episodes before truncating keeps max_episodes meaning "N episodes that are actually
            # trained", rather than "N episodes, some of which the curated set then throws away".
            matched = [r for r in rows if r["episode_id"] in filter_dict]
            if not matched:
                raise ValueError(
                    f"{keep_ranges} is non-empty but matches none of the {len(rows)} episodes in the split; "
                    "check the episode ids. Relative ids are used directly; URI keys need an explicit "
                    "filter_dict_source_prefix (or DROID_FILTER_SOURCE_PREFIX)."
                )
            rows = matched
        if max_episodes is not None:
            rows = rows[:max_episodes]

        windows: list[Window] = []
        # Occurrence count per logical window: overlapping curated ranges legitimately list one window twice,
        # and those draws keep one identity and differ only by draw_id.
        draws: dict[tuple[str, str, int], int] = {}
        for r in rows:
            if filter_dict is not None:
                ranges = filter_dict[r["episode_id"]]
                starts = filtered_starts(ranges, r["length"], self.chunk_length, stride)
            else:
                starts = episode_starts(r["length"], self.chunk_length, stride)
            tasks = r["tasks"] or [""]
            task = tasks[0] if isinstance(tasks, list) else str(tasks)
            video = {
                c: (
                    r[f"videos/{CAM_FEATURE[c]}/chunk_index"],
                    r[f"videos/{CAM_FEATURE[c]}/file_index"],
                    float(r[f"videos/{CAM_FEATURE[c]}/from_timestamp"]),
                )
                for c in CAMS
            }
            for start in starts:
                key = (r["_root_role"], r["episode_id"], start)
                draw_id = draws.get(key, 0)
                draws[key] = draw_id + 1
                windows.append(
                    Window(
                        episode_index=r["episode_index"],
                        episode_id=r["episode_id"],
                        root=r["_root"],
                        root_role=r["_root_role"],
                        start=start,
                        task=task,
                        data_chunk=r["data/chunk_index"],
                        data_file=r["data/file_index"],
                        row_from=r["_row_from"],
                        video=video,
                        draw_id=draw_id,
                    )
                )
        if filter_dict is not None:
            self.filter_stats = {"matched_episodes": len(rows), "kept_windows": len(windows)}
            logger.info(
                "keep-ranges filter %s: %d episodes matched, %d windows kept",
                keep_ranges,
                len(rows),
                len(windows),
            )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def sample_id(self, index: int) -> str:
        """Logical identity of the window at ``index``, independent of where the release is mounted.

        ``dataset_revision`` (when set), the logical root role, the relative episode id, the first frame and the
        window length. The physical root path is deliberately absent, so the same release read on two machines
        yields the same ids, and the window length makes ``chunk_length=16`` and ``32`` distinguishable. Repeated
        draws of one window (overlapping curated ranges) share this id; ``draw_id`` separates them.
        """
        window = self.windows[index]
        parts = [
            self.dataset_revision,
            window.root_role,
            window.episode_id,
            str(window.start),
            str(self.chunk_length),
        ]
        return ":".join(part for part in parts if part)

    # ---------- reading ----------

    def _read_frames(self, cam: str, win: Window) -> torch.Tensor:
        chunk, file_index, from_ts = win.video[cam]
        path = win.root / "videos" / CAM_FEATURE[cam] / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        want_ts = from_ts + win.start / self.fps
        frames = []
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            # One mp4 holds several episodes, so the window is located by timestamp rather than by frame index.
            if want_ts > 0:
                c.seek(int(want_ts / float(stream.time_base)), stream=stream)
            for frame in c.decode(stream):
                if frame.time is not None and frame.time + 1e-6 < want_ts:
                    continue
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
                if len(frames) >= self.n_obs:
                    break
        if len(frames) < self.n_obs:  # near the end of an episode, repeat the last frame
            frames += [frames[-1]] * (self.n_obs - len(frames))
        return torch.stack(frames).permute(0, 3, 1, 2)  # [T, C, H, W] uint8

    def _read_actions(self, win: Window):
        path = win.root / "data" / f"chunk-{win.data_chunk:03d}" / f"file-{win.data_file:03d}.parquet"
        tbl = pq.read_table(path, columns=[JOINT_ACTION, GRIPPER_ACTION, STATE_JOINT, STATE_GRIPPER, "episode_index"])
        lo = win.row_from + win.start
        n = tbl.num_rows
        if lo >= n:
            raise IndexError(
                f"Row {lo} is past the {n} rows of {path.name} (episode {win.episode_index}, start {win.start})."
            )
        got = tbl["episode_index"].to_pylist()[lo]
        if got != win.episode_index:
            raise ValueError(
                f"Row mismatch: expected episode {win.episode_index}, but row {lo} of {path.name} belongs to "
                f"episode {got}."
            )
        joint = torch.tensor(tbl[JOINT_ACTION].to_pylist()[lo : lo + self.chunk_length], dtype=torch.float32)
        grip = torch.tensor(tbl[GRIPPER_ACTION].to_pylist()[lo : lo + self.chunk_length], dtype=torch.float32)
        if grip.ndim == 1:
            grip = grip.unsqueeze(-1)
        # The gripper channel is stored inverted in this release of the dataset. Upstream carries the fact as
        # IS_GRIPPER_ACTION_FLIPPED, True for every droid_plus_lerobot_* version, and applies 1 - g to both the
        # action gripper and the initial-state gripper. Measured on episode 26827 window 0: upstream reports 1.0
        # for all 33 steps where the raw column gives 0.0. Without this the action head learns an inverted
        # open/close command.
        grip = 1.0 - grip
        # Robot state at the first frame of the window, in the same joint(7) + gripper(1) space as the actions.
        s_joint = torch.tensor(tbl[STATE_JOINT].to_pylist()[lo], dtype=torch.float32)
        s_grip = 1.0 - torch.tensor(tbl[STATE_GRIPPER].to_pylist()[lo], dtype=torch.float32)
        state = torch.cat([s_joint.reshape(-1), s_grip.reshape(-1)], dim=-1)
        return torch.cat([joint, grip], dim=-1), state  # [chunk, 8], [8]

    def __getitem__(self, idx: int) -> dict:
        win = self.windows[idx]
        # [0, 1], not [0, 255], because that is the range upstream composes in: LeRobotDataset hands out float
        # frames already divided by 255, and the composition and augmentation run on those values. The range
        # matters at the end of this method, where the round trip back to uint8 is what upstream quantises.
        views = {c: self._read_frames(c, win).float() / 255.0 for c in CAMS}
        if self.augment:
            views = self._augment(views)
        comp = compose_concat_view(views["wrist"], views["left"], views["right"])
        th, tw, aspect = closest_canvas(*comp.shape[-2:], resolution=self.resolution)
        # Quantise before normalising, because upstream does: its dataset ends at
        # (v * 255).clamp(0, 255).to(torch.uint8) and the model later applies x.float() / 127.5 - 1.0. Note that
        # .to(uint8) truncates rather than rounds, so the error it introduces is biased by -0.5 LSB, about
        # 3.9e-03 in [-1, 1] units. Staying in float here would be more accurate and less aligned.
        # Reproduce upstream byte for byte: _convert_video ends at (v * 255).clamp(0, 255).to(torch.uint8) on
        # values that were divided by 255 on the way in. The round trip is not a no-op -- 255.0 * (x / 255.0)
        # in float32 frequently lands just below the integer x, and .to(uint8) truncates rather than rounds, so
        # the result is x - 1. Measured against upstream on one window: without the round trip our pixels are
        # higher by exactly one level on 73.9% of elements and two levels on 3.5%, never lower, relL2 1.16e-02.
        comp = (comp * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        comp, content_hw = resize_and_reflect_pad(comp.float(), th, tw)
        comp = comp / 127.5 - 1.0  # image_mean = image_std = 0.5

        action, state = self._read_actions(win)
        # DROID stores several human annotations of one episode joined by " | " inside a single string, so the
        # multiplicity is invisible from the length of the ``tasks`` list -- every episode in this release has
        # exactly one element. Upstream splits that string and draws one annotation per *draw*
        # (droid_lerobot_dataset.__getitem__), which is a text-side augmentation: the same window is described a
        # different way each time it is visited. Passing the raw string through instead hands the model three
        # instructions glued with pipe characters, a sentence no annotator wrote. The split is unconditional here
        # while upstream gates it on HAS_MULTI_LANGUAGE_ANNOTATIONS[version]; that is the same behaviour for every
        # listed version, because the ones with the flag off contain no " | " and the split is then a no-op.
        # Pinning the draw for a comparison run is the caller's business, not this method's.
        task = random.choice(win.task.split(" | "))
        # The prompt's idle_frame field counts idle frames in the action chunk the prompt describes, and that chunk
        # is the one with the initial state row prepended. packing.pack_one does the prepend, so do it here too
        # rather than counting on 32 rows and disagreeing with upstream by one frame.
        full_action = torch.cat([state.reshape(1, -1).to(action.dtype), action], dim=0)
        return {
            "video": comp.permute(1, 0, 2, 3),  # [C, T, H, W]; the VAE encodes it in the training loop
            "action": action,  # [chunk_length, 8], unnormalized
            "state": state,  # [8]
            "task": task,
            "idle_frames": count_idle_frames(full_action, self.fps),
            "additional_view_description": CONCAT_VIEW_DESCRIPTION,
            "episode_index": win.episode_index,
            "sample_id": self.sample_id(idx),
            "draw_id": win.draw_id,
            "start": win.start,
            "canvas_hw": torch.tensor([th, tw]),
            "content_hw": torch.tensor(content_hw),
            "aspect_key": aspect,
        }

    def _augment(self, views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Upstream's augmentation: the same torchvision v2 transforms, in the same order, on the same tensor.

        ``droid_lerobot_dataset.py`` builds

            T.Compose([
                T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                T.Resize((h, w), antialias=True),
                T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            ])

        and calls it once on ``torch.cat([wrist, left, right], dim=0)``. Both details matter and an earlier
        hand-rolled version got each of them wrong:

        - **One call on the concatenation, not three calls in a loop.** A v2 transform samples its parameters
          once per call and applies them to everything it is handed, so upstream's three views share one crop
          offset and one set of colour factors. Drawing per view, as the loop did, decorrelates them.
        - **A fixed 0.95 crop, not a random scale.** The loop drew ``random.uniform(0.9, 1.0)``, so the crop
          size itself varied; upstream's varies only in position.
        - ``T.Resize(antialias=True)`` rather than a bare bilinear ``F.interpolate``.
        - ``ColorJitter`` at all, which the loop omitted entirely.

        ``jitter_after_compose`` defaults to False and this recipe does not override it, so the jitter belongs
        inside the Compose, on the views, rather than on the composite.

        Args:
            views (dict[str, torch.Tensor]): ``{camera: [T, C, H, W]}``, in :data:`CAMS` order.

        Returns:
            dict[str, torch.Tensor]: The same keys, augmented together.
        """
        names = list(views)
        per_view = views[names[0]].shape[0]
        stacked = torch.cat([views[name] for name in names], dim=0)  # [3T, C, H, W]
        if self._augmentor is None:
            h, w = stacked.shape[-2:]
            self._augmentor = T.Compose(
                [
                    T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                    T.Resize((h, w), antialias=True),
                    T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                ]
            )
        out = self._augmentor(stacked)
        return {name: out[i * per_view : (i + 1) * per_view] for i, name in enumerate(names)}


def collate(batch: list[dict]) -> dict:
    """Stack the tensor fields and keep the rest as lists.

    Videos of different aspect ratios land on different canvases, so a batch is only stackable when its samples
    share one canvas. Strings are never stacked.

    The key list is derived from the sample rather than written out. Writing it out is what dropped
    ``idle_frames`` and ``additional_view_description``: ``__getitem__`` produced them, this function did not
    carry them, and ``build_action_json_prompt`` treats both as optional, so the prompt silently lost the view
    description and the idle-frame count -- 104 tokens where upstream, on the same window, emits 149.
    """
    keys = batch[0].keys()
    # Symmetric difference, not ``keys - b``: a later sample can also carry a field the first one lacks, and that
    # extra field would otherwise be dropped silently.
    mismatch = [k for b in batch for k in set(keys).symmetric_difference(b)]
    if mismatch:
        # "same keys" keeps the upstream contract and its test; "schema mismatch" says what the symmetric
        # difference actually means -- a later sample may carry an extra field, not only miss one.
        raise ValueError(
            "The samples of a batch do not carry the same keys (schema mismatch); "
            f"mismatched keys: {sorted(set(mismatch))}"
        )
    out: dict = {}
    for key in keys:
        values = [b[key] for b in batch]
        out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
    return out

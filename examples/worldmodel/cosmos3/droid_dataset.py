"""Map-style DROID action-policy dataset for the Cosmos 3 SFT recipe.

Follows the preprocessing of ``action_policy_droid_nano`` in nvidia/cosmos-framework:
``concat_view`` (wrist on top, the two shoulder views side by side at half width below), resized onto the closest
predefined canvas, then reflection-padded at the bottom and right. ``chunk_length=32`` gives 33 observation frames
and 32 actions, at 15 fps, in the unnormalized 8-dimensional ``joint_pos`` action space.

Two design decisions worth stating, because both are easy to get wrong in the other direction:

1. **Samples carry pixels, not latents.** The VAE runs inside the training loop, as it does upstream, because a
   DataLoader worker process cannot touch an NPU device. Image augmentation also has to happen before the VAE, so
   encoding ahead of time would freeze the augmentation into the cached latents.

2. **Index order is episode-shuffled and sequential within an episode**, to be used with a ``shuffle=False``
   sampler. Upstream explains why: purely sequential sampling hands one rank a run of overlapping windows, so the
   global batch covers about one episode and the gradient norm becomes unstable, while a random sampler decorrelates
   the batch at the cost of random-access I/O. Shuffling at index-construction time and leaving the sampler
   sequential gives both the decorrelation and the sequential reads.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

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
    """One training window: 33 observation frames and 32 actions starting at ``start`` inside an episode."""

    episode_index: int
    start: int  # first frame, relative to the episode
    task: str
    data_chunk: int
    data_file: int
    row_from: int  # first row of this episode **within its own parquet file**
    video: dict[str, tuple[int, int, float]]  # cam -> (chunk, file, from_timestamp)


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
        root (str | Path): Dataset root, the directory holding ``meta/``, ``data/`` and ``videos/``.
        chunk_length (int, optional): Number of actions per window; the window holds ``chunk_length + 1``
            observation frames. Defaults to 32.
        fps (float, optional): Frame rate the video timestamps are resolved against. Defaults to 15.0.
        resolution (str, optional): Key into :data:`RES_TABLE`. Defaults to "480".
        sample_stride (int, optional): Step between window starts inside an episode. Defaults to 1.
        episode_shuffle_seed (int, optional): Seed of the episode-order shuffle. Defaults to 42.
        max_episodes (int | None, optional): Index only the first N episodes after the shuffle, for smoke runs.
            Defaults to None.
        augment (bool, optional): Apply the image augmentation. Defaults to False.
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
    ):
        self.root = Path(root)
        self.chunk_length = chunk_length
        self.n_obs = chunk_length + 1
        self.fps = fps
        self.resolution = resolution
        self.augment = augment
        # Upstream DROIDLeRobotDataset defaults: split_seed=42 (passed as episode_shuffle_seed here, since the
        # same permutation does both jobs) and split_val_ratio=0.03.
        self.split = split
        self.split_val_ratio = split_val_ratio
        # Built lazily, as upstream builds it: the crop size is a fraction of the first batch's frame size.
        self._augmentor: T.Compose | None = None
        self._info = json.loads((self.root / "meta/info.json").read_text())
        self.windows = self._build_index(sample_stride, episode_shuffle_seed, max_episodes)

    # ---------- index ----------

    def _build_index(self, stride, seed, max_episodes) -> list[Window]:
        cols = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index", "dataset_from_index"]
        for cam in CAMS:
            f = CAM_FEATURE[cam]
            cols += [f"videos/{f}/chunk_index", f"videos/{f}/file_index", f"videos/{f}/from_timestamp"]
        rows: list[dict] = []
        for p in sorted((self.root / "meta/episodes").rglob("*.parquet")):
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

        # Episode order AND the train/val split come from one torch.randperm, because that is what upstream
        # does: split_episode_ids draws randperm(total_episodes, seed) and returns episode_ids[num_val:] for
        # train, and build_episode_spans then walks the index in exactly that order. So the permutation is both
        # the holdout and the shuffle.
        #
        # Two things were wrong here before. We used random.Random(seed).shuffle, a different algorithm from
        # torch.randperm, so our episode order differed from upstream even at the same seed. And we kept every
        # episode: upstream reserves split_val_ratio=0.03 of them, which on success/ is
        # round(57639 * 0.03) = 1729 episodes, leaving the 55910 its log reports against our 57639. Training on
        # all of them means training on upstream published validation set, so no evaluation of ours was
        # comparable to theirs.
        total_episodes = int(self._info.get("total_episodes", len(rows)))
        order = torch.randperm(total_episodes, generator=torch.Generator().manual_seed(seed)).tolist()
        num_val = int(round(total_episodes * self.split_val_ratio))
        if self.split == "train":
            keep_ids = order[num_val:]
        elif self.split == "val":
            keep_ids = order[:num_val]
        elif self.split == "full":
            keep_ids = order
        else:
            raise ValueError(f"split must be train, val or full; got {self.split!r}")
        by_id = {int(r["episode_index"]): r for r in rows}
        rows = [by_id[i] for i in keep_ids if i in by_id]
        if max_episodes is not None:
            rows = rows[:max_episodes]

        windows: list[Window] = []
        for r in rows:
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
            usable = r["length"] - self.n_obs
            if usable < 0:
                # Too short to fill one window. max(usable, 0) + 1 used to emit a window anyway, which then
                # read past the end of its episode; upstream requires stop - start - chunk_length > 0 and drops
                # the episode. 55 episodes in success/ are this short, which is exactly the gap between our
                # 16,847,949 windows and the 16,847,894 the metadata implies.
                continue
            for start in range(0, usable + 1, stride):
                local_from = r["dataset_from_index"] - base[(r["data/chunk_index"], r["data/file_index"])]
                windows.append(
                    Window(
                        r["episode_index"],
                        start,
                        task,
                        r["data/chunk_index"],
                        r["data/file_index"],
                        local_from,
                        video,
                    )
                )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    # ---------- reading ----------

    def _read_frames(self, cam: str, win: Window) -> torch.Tensor:
        chunk, file_index, from_ts = win.video[cam]
        path = self.root / "videos" / CAM_FEATURE[cam] / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
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
        path = self.root / "data" / f"chunk-{win.data_chunk:03d}" / f"file-{win.data_file:03d}.parquet"
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
    """
    out: dict = {}
    for key in ("video", "action", "state", "canvas_hw", "content_hw"):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("task", "episode_index", "start", "aspect_key"):
        out[key] = [b[key] for b in batch]
    return out

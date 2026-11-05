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
    scale = min(target_h / h, target_w / w)
    rh, rw = int(scale * h + 0.5), int(scale * w + 0.5)
    if (rh, rw) != (h, w):
        x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
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
    ):
        self.root = Path(root)
        self.chunk_length = chunk_length
        self.n_obs = chunk_length + 1
        self.fps = fps
        self.resolution = resolution
        self.augment = augment
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

        rng = random.Random(seed)
        rng.shuffle(rows)  # shuffle episode order, keep windows sequential inside an episode
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
            for start in range(0, max(usable, 0) + 1, stride):
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
        # Robot state at the first frame of the window, in the same joint(7) + gripper(1) space as the actions.
        s_joint = torch.tensor(tbl[STATE_JOINT].to_pylist()[lo], dtype=torch.float32)
        s_grip = torch.tensor(tbl[STATE_GRIPPER].to_pylist()[lo], dtype=torch.float32)
        state = torch.cat([s_joint.reshape(-1), s_grip.reshape(-1)], dim=-1)
        return torch.cat([joint, grip], dim=-1), state  # [chunk, 8], [8]

    def __getitem__(self, idx: int) -> dict:
        win = self.windows[idx]
        views = {c: self._read_frames(c, win).float() for c in CAMS}
        if self.augment:
            views = self._augment(views)
        comp = compose_concat_view(views["wrist"], views["left"], views["right"])
        th, tw, aspect = closest_canvas(*comp.shape[-2:], resolution=self.resolution)
        comp, content_hw = resize_and_reflect_pad(comp, th, tw)
        comp = comp / 127.5 - 1.0  # image_mean = image_std = 0.5

        action, state = self._read_actions(win)
        return {
            "video": comp.permute(1, 0, 2, 3),  # [C, T, H, W]; the VAE encodes it in the training loop
            "action": action,  # [chunk_length, 8], unnormalized
            "state": state,  # [8]
            "task": win.task,
            "episode_index": win.episode_index,
            "start": win.start,
            "canvas_hw": torch.tensor([th, tw]),
            "content_hw": torch.tensor(content_hw),
            "aspect_key": aspect,
        }

    def _augment(self, views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Random resized crop applied to all three views at once.

        Upstream also applies a ColorJitter whose parameters are shared across the three views. That part is not
        implemented yet, so a run with ``augment=True`` is close to but not identical to the recipe.
        """
        out = {}
        scale = random.uniform(0.9, 1.0)
        for name, v in views.items():
            h, w = v.shape[-2:]
            ch, cw = int(h * scale), int(w * scale)
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            crop = v[..., top : top + ch, left : left + cw]
            out[name] = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
        return out


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

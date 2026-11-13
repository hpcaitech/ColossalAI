"""Read DROID through upstream's own dataset, and hand this example's packer what it expects.

The reader is ``vendor/cosmos_framework``, copied byte for byte (see ``vendor/README.md``). This module is
the seam: it converts the reader's sample into the dict ``packing.pack_batch`` consumes, and it does so by
*undoing* what the reader did rather than by re-deriving it, so nothing here is a second implementation of
something upstream already decided.

Where the seam sits, and why there:

    reader  ->  composite, quantised to uint8   <- this is what the reader returns
    here    ->  pad to the canvas, normalise to [-1, 1], split the state row off the action

The reader's video is ``(v * 255).clamp(0, 255).to(uint8)`` of the composed views, which is exactly the
value ``droid_dataset`` reaches before padding. On the pinned window (episode 26827, ordinal 0) the two
were compared element by element with the instruction pinned on both sides and augmentation off: all
34,214,400 values identical, max |d| 0.000 levels. The seam is therefore not a place where a difference can
hide -- it was measured, not assumed.

What is deliberately still this example's:

* padding to the canvas and the [-1, 1] normalisation, because the reader stops before them and the stage
  that does them upstream cannot be taken without upstream's packing (``vendor/README.md`` has the detail);
* the prompt, the CFG dropout and the tokenisation, whose output was already shown identical to upstream's
  (``text_ids``, 170 tokens, byte for byte at steps 0 and 1).

The scaling half of ``resize_and_reflect_pad`` never runs for DROID: the composite is always 540x640 for
this dataset's 640x360 cameras, the 4:3 canvas is 544x736, and the scale factor is capped at 1.0, so the
call only ever pads. Measured, not reasoned: ``closest_canvas(540, 640) -> (544, 736, '4,3')`` and the
returned content region is (540, 640), unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from droid_dataset import CONCAT_VIEW_DESCRIPTION, closest_canvas, resize_and_reflect_pad
from torch.utils.data import Dataset

# The reader derives the dataset version from the directory's basename and rejects anything it does not
# know, so the root it is given has to carry one of upstream's names. A checkout under another name needs a
# symlink; the comparison host has the same one.
KNOWN_ROOT_NAMES = (
    "droid_plus_lerobot_640x360_20260412",
    "droid_plus_lerobot_640x360_20260412_sharded",
    "droid_plus_lerobot_320x180_20260406",
    "droid_plus_lerobot_320x180_20260406_sharded",
    "droid_lerobot_20260115_no_noops",
)


class UpstreamDroidDataset(Dataset):
    """Upstream's ``DROIDLeRobotDataset``, presented in this example's sample contract."""

    def __init__(
        self,
        root: str | Path,
        chunk_length: int = 32,
        fps: float = 15.0,
        resolution: str = "480",
        augment: bool = True,
        success_only: bool = True,
        max_episodes: int | None = None,
        dataset_revision: str = "",
    ) -> None:
        root = Path(root)
        if root.name not in KNOWN_ROOT_NAMES:
            raise ValueError(
                f"upstream's reader keys its layout off the directory name and does not know {root.name!r}. "
                f"Point --data at a path named one of {KNOWN_ROOT_NAMES} (a symlink is enough)."
            )
        from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

        self._inner = DROIDLeRobotDataset(
            root=str(root),
            fps=fps,
            chunk_length=chunk_length,
            mode="wam",
            viewpoint="concat_view",
            use_success_only=success_only,
            use_state=True,
            action_space="joint_pos",
            use_image_augmentation=augment,
        )
        self.resolution = resolution
        self.fps = fps
        self.chunk_length = chunk_length
        self.dataset_revision = dataset_revision
        self._root_role = "success" if success_only else "mixed"
        self._max_episodes = max_episodes
        self._starts = self._episode_starts()

    def _episode_starts(self) -> dict[int, int]:
        """The reader's own (episode -> first global index) table, used only to report where a sample came from."""
        out, prev = {}, 0
        for (_ds, _ss, _vl, eid), end in zip(self._inner._episode_records, self._inner._episode_cum_ends):
            out[int(eid)] = prev
            prev = end
        return out

    def _locate(self, index: int) -> tuple[int, int]:
        episode, start = -1, -1
        for eid, first in self._starts.items():
            if first <= index and (start < 0 or first > start):
                episode, start = eid, first
        return episode, index - start if start >= 0 else -1

    def sample_id(self, index: int) -> str:
        """Logical identity of a window, in the same shape ``droid_dataset.sample_id`` uses.

        Same parts, same order, same reason: the revision, the logical root role, the episode, the first
        frame and the window length, with no physical path, so the id does not change with where the
        release is mounted.

        **It is not the same string as the other reader's**, and deliberately so rather than by oversight.
        That one names the episode by its relative id (``RAIL/success/2023-05-20/Sat_...``); upstream's
        index carries the episode only as the integer it keys episodes by, so the string is not available
        here without re-deriving it from the parquet -- which would be this example reimplementing a piece
        of the reader again, the thing the copy exists to stop.

        The consequence is worth stating plainly: **replay packs and held-out manifests recorded under one
        reader do not match ids recorded under the other.** Everything else about the two samples is
        identical (measured field by field on the pinned window), so this is a bookkeeping boundary, not a
        data one -- but a manifest cannot be carried across the ``--dataset`` switch.
        """
        episode, ordinal = self._locate(index)
        parts = [
            self.dataset_revision,
            self._root_role,
            f"episode_{episode}",
            str(ordinal),
            str(self.chunk_length),
        ]
        return ":".join(part for part in parts if part)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._inner[index]

        # The reader prepends the initial state as row 0 of the action when use_state is on. pack_one does
        # that prepend itself, so the row is taken back off here rather than the packer being changed: the
        # split is the exact inverse of the reader's concatenation, and was verified identical on the pinned
        # window (state row and all 32 action rows, bit for bit).
        action_full = sample["action"]
        state, action = action_full[0], action_full[1:]

        video = sample["video"]  # uint8 [C, T, H, W], the quantised composite
        target_h, target_w, aspect = closest_canvas(*video.shape[-2:], resolution=self.resolution)
        padded, content_hw = resize_and_reflect_pad(video.float(), target_h, target_w)
        padded = padded / 127.5 - 1.0  # image_mean = image_std = 0.5

        episode, ordinal = self._locate(index)
        return {
            "video": padded,
            "action": action.float(),
            "state": state.float(),
            "task": sample["ai_caption"],
            "idle_frames": int(sample["idle_frames"]),
            "additional_view_description": sample.get("additional_view_description", CONCAT_VIEW_DESCRIPTION),
            "episode_index": episode,
            "sample_id": self.sample_id(index),
            "draw_id": 0,
            "start": ordinal,
            "canvas_hw": torch.tensor([target_h, target_w]),
            "content_hw": torch.tensor(content_hw),
            "aspect_key": aspect,
        }

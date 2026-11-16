"""How many frames does one window's timestamp seek actually decode?

``DroidActionPolicyDataset._read_frames`` seeks to ``from_timestamp + start / fps`` and then decodes forward,
keeping 33 frames. If the seek lands far before the target, the decoder walks forward through the file and the
sample costs tens of seconds -- which is what the periodic ~199 s step stall looks like: one rank's batch waits on
those samples while every other rank has already queued its data.

This script reproduces the seek on the windows the trainer reports as slow and counts frames decoded vs kept, so the
cost can be attributed to the seek or ruled out.

    python seek_cost.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import av

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.worldmodel.cosmos3.droid_dataset import CAM_FEATURE, DroidActionPolicyDataset

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"
FPS = 15.0
N_OBS = 33


def probe(dataset, index: int, camera: str = "wrist") -> None:
    window = dataset.windows[index]
    chunk, file_index, from_ts = window.video[camera]
    path = window.root / "videos" / CAM_FEATURE[camera] / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    want_ts = from_ts + window.start / FPS

    started = time.perf_counter()
    decoded = 0
    kept = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if want_ts > 0:
            container.seek(int(want_ts / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            decoded += 1
            if frame.time is not None and frame.time + 1e-6 < want_ts:
                continue
            kept += 1
            if kept >= N_OBS:
                break
    elapsed = time.perf_counter() - started
    print(
        f"index={index} episode={window.episode_index} start={window.start} want_ts={want_ts:.2f}s "
        f"kept={kept} decoded={decoded} ratio={decoded / max(kept, 1):.1f}x time={elapsed:.2f}s",
        flush=True,
    )


dataset = DroidActionPolicyDataset(DATA, chunk_length=32, fps=FPS, resolution="480", augment=False, max_episodes=64)
print(f"windows={len(dataset)}")

# A spread rather than a sweep: the windows inside one episode, one at the start of an episode, and one deep in a
# long episode, which is where a bad seek would cost the most.
for index in [0, 1, 2, 11, 50, 100, 400, 900, 1500, 2000]:
    if index < len(dataset):
        probe(dataset, index)

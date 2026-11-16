"""Does PyAV's decoder thread_type speed up the frame decode, and are the frames identical?

The data path decodes 33 frames from each of three camera mp4s per sample, and ``_read_frames`` never sets
``thread_type``, so the decoder runs single-threaded. Frame-level threading should be a pure win -- the decoded
pixels are what they are -- but the only way to be sure is to compare the frames.

    python scripts/probe_decode_threads.py [--episodes 3]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import av
import torch

ROOT = Path("/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success")
FRAMES = 33


def decode(path: Path, thread_type: str | None, thread_count: int = 0) -> list[torch.Tensor]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if thread_type is not None:
            stream.thread_type = thread_type
        if thread_count:
            stream.codec_context.thread_count = thread_count
        for frame in container.decode(stream):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if len(frames) >= FRAMES:
                break
    return frames


def bench(fn, iters=5):
    fn()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - started) / iters * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    files = sorted(ROOT.glob("videos/*/chunk-*/file-*.mp4"))[: args.episodes]
    if not files:
        raise SystemExit(f"no mp4 under {ROOT}")
    print(f"[decode] {len(files)} file(s), {FRAMES} frames each")

    for path in files:
        baseline = decode(path, None)
        line = [f"  {path.name:<22}"]
        for label, thread_type, count in (("none", None, 0), ("AUTO", "AUTO", 0), ("AUTO/4", "AUTO", 4), ("FRAME/4", "FRAME", 4)):
            ms = bench(lambda t=thread_type, c=count: decode(path, t, c))
            if thread_type is None:
                equal, diff = True, 0
            else:
                other = decode(path, thread_type, count)
                equal = all(torch.equal(a, b) for a, b in zip(baseline, other)) and len(baseline) == len(other)
                diff = sum(int((a != b).sum()) for a, b in zip(baseline, other))
            mark = "=" if equal else f"!{diff}"
            line.append(f"{label} {ms:6.0f}ms{mark}")
        print("".join(line), flush=True)


if __name__ == "__main__":
    main()

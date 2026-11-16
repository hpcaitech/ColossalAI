"""Who imports torch_npu inside a DataLoader worker, and does it happen before or after torch.compile?

Measured symptom: with ``COSMOS3_FAST_JITTER=1`` every DataLoader worker holds an NPU context on card 0
(32 workers x 110 MB = 3.5 GB, all on rank 0's card; with the jitter off there are none). Setting
``ASCEND_RT_VISIBLE_DEVICES=""`` before ``torch.compile`` did not help, so the context is created before that call.

This runs one spawned child -- the same start method the DataLoader uses -- and reports, at each step:
whether ``torch_npu`` is in ``sys.modules``, and if it is, where it came from (first import stack).

    python scripts/probe_worker_npu_import.py
"""

from __future__ import annotations

import os
import sys
import traceback
import torch
import torch.multiprocessing as mp


def child(_rank: int) -> None:
    report = []

    def state(label: str) -> None:
        loaded = "torch_npu" in sys.modules
        report.append(f"{label:<34} torch_npu loaded: {loaded}")
        if loaded and not any("first import" in line for line in report):
            import torch_npu  # noqa: F401  (record where it was first imported)

            report.append(f"{'':<34} first import: {getattr(torch_npu, '__file__', '?')}")

    state("at child start")

    # What the dataset does on the first sample.
    import torchvision.transforms.v2 as T

    state("after torchvision import")

    from examples.worldmodel.cosmos3.fast_color_jitter import FastColorJitter

    state("after fast_color_jitter import")

    jitter = FastColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
    x = torch.rand(4, 3, 64, 64)
    jitter(x)
    state("after first jitter call (compile)")

    # Does a second call change anything?
    jitter(x)
    state("after second jitter call")

    print("\n".join(report), flush=True)


def main() -> None:
    mp.set_start_method("spawn", force=True)
    process = mp.Process(target=child, args=(0,))
    process.start()
    process.join()


if __name__ == "__main__":
    main()

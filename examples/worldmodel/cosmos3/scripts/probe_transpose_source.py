"""Where do the encoder's 61 transposed copies come from?

The kernel profile has ``aclnnInplaceCopy_TransposeAiCore_Transpose`` 61 times per encode (0.109 s, 4.4%),
which is what a ``reshape``/``contiguous`` on a permuted view lowers to. ``--with-stack`` profiling does not
produce a CSV on this stack, so this probe patches ``Tensor.reshape`` / ``Tensor.contiguous`` in Python and
records the call site of every materialisation instead.

    python scripts/probe_transpose_source.py --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID
"""

from __future__ import annotations

import argparse
import traceback
from collections import Counter

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    enable_fast_encoder(vae)

    hits = Counter()
    original_contiguous = torch.Tensor.contiguous
    original_reshape = torch.Tensor.reshape

    def site():
        # The last frame outside torch internals is the model code we are looking for.
        stack = traceback.extract_stack()[:-2]
        for frame in reversed(stack):
            if "/torch/" not in frame.filename and "site-packages/torch" not in frame.filename:
                return f"{frame.filename.split('/examples/')[-1]}:{frame.lineno} in {frame.name}"
        return "?"

    def patched_contiguous(self, *a, **kw):
        if not self.is_contiguous():
            hits[("contiguous", site())] += 1
        return original_contiguous(self, *a, **kw)

    def patched_reshape(self, *shape):
        if not self.is_contiguous():
            hits[("reshape", site())] += 1
        return original_reshape(self, *shape)

    torch.Tensor.contiguous = patched_contiguous
    torch.Tensor.reshape = patched_reshape

    x = torch.randn(args.batch, 3, 33, 544, 656, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        vae.encode(x)

    torch.Tensor.contiguous = original_contiguous
    torch.Tensor.reshape = original_reshape

    print(f"[hits] {sum(hits.values())} materialised views over one encode at batch {args.batch}")
    for (kind, where), count in hits.most_common(12):
        print(f"  {count:4d}  {kind:10s} {where}")
    print("[device kernels] a transposed copy is one Transpose kernel; scale by the profile's 61 per encode")


if __name__ == "__main__":
    main()

"""Can the VAE encode and a cube-bound workload actually run at the same time on this stack?

The cross-step overlap design (VAE of batch k+1 under the backward of step k) only pays if the two use different
engines concurrently. The step profile says the summed kernel time is 1.4x the wall, so *something* overlaps -- but
whether a conv/vector-heavy encode can hide under a cube-heavy backward is a different question, and guide 08 warns
that two streams contend for HBM/L2.

This measures it directly: the real VAE encode (A) against a matmul chain of comparable duration (B), alone and on
two streams. If concurrent ~= max(A, B) the overlap is worth building; if concurrent ~= A + B it is not.

    python scripts/probe_stream_overlap.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse

import torch


def bench(fn, iters=5):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--matmul-n", type=int, default=8192, help="Square matmul size for the cube workload.")
    parser.add_argument("--matmul-reps", type=int, default=100, help="Matmuls per call, to match the encode.")
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan

    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder
    from examples.worldmodel.cosmos3.vae_norm_kernel import enable_kernel_rmsnorm

    device = torch.device("npu:0")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint + "/vae", torch_dtype=torch.bfloat16).eval().to(device)
    vae.requires_grad_(False)
    enable_fast_encoder(vae)
    enable_kernel_rmsnorm(vae)

    video = torch.randn(args.batch, 3, 33, 544, 656, device=device, dtype=torch.bfloat16)
    n = args.matmul_n
    reps = args.matmul_reps
    left = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    right = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    out = torch.empty(n, n, device=device, dtype=torch.bfloat16)
    side = torch.npu.Stream()

    def encode():
        with torch.no_grad():
            vae.encode(video)

    def matmul():
        for _ in range(reps):
            torch.matmul(left, right, out=out)

    ms_encode = bench(encode, args.iters)
    ms_matmul = bench(matmul, args.iters)
    ms_sequential = bench(lambda: (encode(), matmul()), args.iters)

    def concurrent():
        done = torch.npu.Event()
        current = torch.npu.current_stream()
        side.wait_stream(current)
        with torch.npu.stream(side):
            encode()
        done.record(side)
        matmul()
        current.wait_event(done)

    ms_concurrent = bench(concurrent, args.iters)

    print(f"[overlap] encode {ms_encode:.2f} ms | matmul({n})x{reps} {ms_matmul:.2f} ms | sequential {ms_sequential:.2f} ms")
    print(f"[overlap] two streams {ms_concurrent:.2f} ms  "
          f"(sum would be {ms_encode + ms_matmul:.2f}, max would be {max(ms_encode, ms_matmul):.2f})")
    saved = ms_sequential - ms_concurrent
    print(f"[overlap] {'OVERLAPS' if ms_concurrent < 0.9 * ms_sequential else 'SERIALISES'}: "
          f"{saved:.2f} ms saved of {min(ms_encode, ms_matmul):.2f} ms of hideable work "
          f"({100 * saved / min(ms_encode, ms_matmul):.0f}% of the smaller side)")


if __name__ == "__main__":
    main()

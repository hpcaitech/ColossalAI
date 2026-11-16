"""What does this node's HCCL actually deliver, and is the training's reduce-scatter near it?

The profiled step shows ~10.5 s of hcom with no overlap against compute -- 56 reduce-scatters at ~0.17 s each.
Per bucket that is 25M trainable params, so ~50 MB bf16 or ~100 MB fp32; 0.17 s for 50 MB is 300 MB/s. This
probe measures the fabric directly at the same message sizes and at 1 GB, so the next step is a decision:
fix the transport, reduce the volume, or overlap the communication.

    torchrun --nproc_per_node=8 scripts/bench_hccl.py [--mb 512]
"""

import argparse
import time

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mb", type=int, default=512, help="Payload size in MiB per buffer.")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    device = torch.device(f"npu:{rank % 8}")
    torch.npu.set_device(device)

    for mb in (args.mb // 8, args.mb // 2, args.mb):
        count = mb * 1024 * 1024 // 2  # bf16 elements
        buf = torch.ones(count, device=device, dtype=torch.bfloat16)
        out = torch.empty(count // dist.get_world_size(), device=device, dtype=torch.bfloat16)
        for name, fn in (
            ("all_reduce", lambda: dist.all_reduce(buf)),
            ("reduce_scatter", lambda: dist.reduce_scatter_tensor(out, buf)),
        ):
            for _ in range(3):
                fn()
            torch.npu.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(args.iters):
                fn()
            torch.npu.synchronize()
            dist.barrier()
            elapsed = (time.perf_counter() - start) / args.iters
            # reduce_scatter moves roughly (n-1)/n of the buffer per rank; report the payload rate either way
            gbps = mb / 1024 / elapsed
            if rank == 0:
                print(f"[hccl] {name:14s} {mb:5d} MiB  {elapsed * 1e3:8.2f} ms  {gbps:7.2f} GiB/s", flush=True)
        del buf, out
        torch.npu.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

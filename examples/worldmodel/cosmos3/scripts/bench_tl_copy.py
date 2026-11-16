"""What copy bandwidth can this TileLang build reach on the 910B2?

The fused channel-first RMSNorm is only worth writing if a TileLang kernel can move memory at a large fraction of
HBM bandwidth: the CANN chain it replaces already runs at ~700 GB/s of effective traffic (six full-tensor passes).
The earlier uint8 kernel reached only ~5 GB/s, which would doom the fusion, but it was a flat kernel with a
50k-block grid and 21 KB per block. This probe separates the variables: grid-vs-persistent, block size, and
threads per block, against ``torch.copy_`` and ``torch.mul`` as the aclnn reference.

    python scripts/bench_tl_copy.py
"""

import tilelang
import tilelang.language as T
import torch

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CORES = 48
N = CORES * 16384 * 128  # 100,663,296 bf16 elements = 201 MB per direction


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def copy_grid(N: int, block: int, threads: int):
    total = N // block

    @T.prim_func
    def main(src: T.Tensor((N,), "bfloat16"), dst: T.Tensor((N,), "bfloat16")):
        with T.Kernel(total, threads=threads, is_npu=True) as cid:
            s = T.alloc_shared((block,), "bfloat16")
            T.copy(src[cid * block : (cid + 1) * block], s)
            T.copy(s, dst[cid * block : (cid + 1) * block])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def copy_persist(N: int, block: int, cores: int, threads: int):
    iters = N // (block * cores)

    @T.prim_func
    def main(src: T.Tensor((N,), "bfloat16"), dst: T.Tensor((N,), "bfloat16")):
        with T.Kernel(cores, threads=threads, is_npu=True) as cid:
            s = T.alloc_shared((block,), "bfloat16")
            for it in T.serial(iters):
                off = (it * cores + cid) * block
                T.copy(src[off : off + block], s)
                T.copy(s, dst[off : off + block])

    return main


def bench(fn, iters=20):
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
    ms = start.elapsed_time(end) / iters
    return ms


def report(tag, ms):
    gbps = 2 * N * 2 / (ms * 1e6)
    print(f"{tag:44s} {ms:8.3f} ms  {gbps:7.1f} GB/s", flush=True)


def main():
    device = torch.device("npu:0")
    src = torch.randn(N, device=device, dtype=torch.bfloat16)
    dst = torch.empty_like(src)

    report("torch copy_ (aclnn)", bench(lambda: dst.copy_(src)))
    report("torch mul out= (aclnn)", bench(lambda: torch.mul(src, 2.0, out=dst)))
    report("torch add out= (aclnn)", bench(lambda: torch.add(src, 1.0, out=dst)))

    for block in (8192, 32768, 131072):
        for threads in (1, 2, 4):
            try:
                k = copy_grid(N, block, threads)
                k(src)
                torch.npu.synchronize()
                report(f"tl grid block={block} threads={threads}", bench(lambda: k(src)))
            except Exception as exc:  # noqa: BLE001 - a failing variant is data, not an error
                print(
                    f"tl grid block={block} threads={threads}: FAILED {type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )

    for block in (8192, 32768, 131072):
        for threads in (1, 2):
            try:
                k = copy_persist(N, block, CORES, threads)
                k(src)
                torch.npu.synchronize()
                report(f"tl persist block={block} threads={threads}", bench(lambda: k(src)))
            except Exception as exc:  # noqa: BLE001
                print(
                    f"tl persist block={block} threads={threads}: FAILED {type(exc).__name__}: {str(exc)[:120]}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

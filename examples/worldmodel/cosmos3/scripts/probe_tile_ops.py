"""Which ascend_tile vector ops actually work in this TileLang build, and do they round like torch?

The fused norm needs bf16->fp32 cast, fp32 vector multiply/add, fp32 scalar multiply, and fp32->bf16 cast. The
build has a history of the last one producing zeros with ``CAST_NONE``, so each cast mode is compared against
``torch.Tensor.to`` before it goes into a kernel.

    python scripts/probe_tile_ops.py
"""

import torch
import tilelang
import tilelang.language as T

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

N = 2048
MODES = ("CAST_NONE", "CAST_RINT", "CAST_ROUND", "CAST_FLOOR", "CAST_TRUNC")


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def bf16_to_fp32(mode: str):
    @T.prim_func
    def main(src: T.Tensor((N,), "bfloat16"), dst: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1, is_npu=True) as _:
            s = T.alloc_ub((N,), "bfloat16")
            d = T.alloc_ub((N,), "float32")
            T.copy(src, s)
            T.tile.cast(d, s, mode, N)
            T.copy(d, dst)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def fp32_to_bf16(mode: str):
    @T.prim_func
    def main(src: T.Tensor((N,), "float32"), dst: T.Tensor((N,), "bfloat16")):
        with T.Kernel(1, threads=1, is_npu=True) as _:
            s = T.alloc_ub((N,), "float32")
            d = T.alloc_ub((N,), "bfloat16")
            T.copy(src, s)
            T.tile.cast(d, s, mode, N)
            T.copy(d, dst)

    return main


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def vector_math():
    """dst = ((a * b) * 2.5 + 1.0) elementwise on fp32 UB buffers, plus a fill."""
    @T.prim_func
    def main(a: T.Tensor((N,), "float32"), b: T.Tensor((N,), "float32"), dst: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1, is_npu=True) as _:
            a_ub = T.alloc_ub((N,), "float32")
            b_ub = T.alloc_ub((N,), "float32")
            d_ub = T.alloc_ub((N,), "float32")
            T.copy(a, a_ub)
            T.copy(b, b_ub)
            T.tile.fill(d_ub, 0.0)
            T.tile.mul(d_ub, a_ub, b_ub)
            T.tile.mul(d_ub, d_ub, 2.5)
            T.tile.add(d_ub, d_ub, 1.0)
            T.copy(d_ub, dst)

    return main


def main():
    device = torch.device("npu:0")
    src_bf16 = torch.randn(N, device=device, dtype=torch.bfloat16)
    src_fp32 = torch.randn(N, device=device, dtype=torch.float32)

    print("[bf16 -> fp32]")
    for mode in MODES:
        try:
            out = bf16_to_fp32(mode)(src_bf16)
            ref = src_bf16.float()
            print(f"  {mode:11s} maxabs {float((out - ref).abs().max()):.3e}  equal {bool(torch.equal(out, ref))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mode:11s} FAILED {type(exc).__name__}: {str(exc)[:100]}")

    print("[fp32 -> bf16]")
    for mode in MODES:
        try:
            out = fp32_to_bf16(mode)(src_fp32)
            ref = src_fp32.to(torch.bfloat16)
            same = float((out.float() - ref.float()).abs().max())
            zeros = int((out == 0).sum().item())
            print(f"  {mode:11s} maxabs-vs-torch {same:.3e}  zeros {zeros}  equal {bool(torch.equal(out, ref))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mode:11s} FAILED {type(exc).__name__}: {str(exc)[:100]}")

    print("[vector math]")
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    try:
        out = vector_math()(a, b)
        ref = (a * b) * 2.5 + 1.0
        print(f"  mul/muls/adds maxabs {float((out - ref).abs().max()):.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  vector math FAILED {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()

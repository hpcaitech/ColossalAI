"""Isolate which step of the uint8 front end zeroes the output: copy, CANN Cast uint8->fp16, or the fp32 normalise."""

import tilelang
import tilelang.language as T
import torch

PASS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
BLOCK = 4096
M = 16384


@tilelang.jit(out_idx=[-1], pass_configs=PASS)
def _pass(M: int, block: int = BLOCK):
    @T.prim_func
    def main(src: T.Tensor((M,), "uint8"), dst: T.Tensor((M,), "uint8")):
        with T.Kernel(M // block, threads=2, is_npu=True) as cid:
            s = T.alloc_shared((block,), "uint8")
            T.copy(src[cid * block : (cid + 1) * block], s)
            T.copy(s, dst[cid * block : (cid + 1) * block])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=PASS)
def _cast(M: int, block: int = BLOCK):
    @T.prim_func
    def main(src: T.Tensor((M,), "uint8"), dst: T.Tensor((M,), "float16")):
        with T.Kernel(M // block, threads=2, is_npu=True) as cid:
            s = T.alloc_shared((block,), "uint8")
            h = T.alloc_shared((block,), "float16")
            T.copy(src[cid * block : (cid + 1) * block], s)
            T.tile.cast(h, s, "CAST_NONE", block)
            T.copy(h, dst[cid * block : (cid + 1) * block])

    return main


@tilelang.jit(out_idx=[-1], pass_configs=PASS)
def _norm(M: int, block: int = BLOCK):
    @T.prim_func
    def main(src: T.Tensor((M,), "uint8"), dst: T.Tensor((M,), "float32")):
        with T.Kernel(M // block, threads=2, is_npu=True) as cid:
            s = T.alloc_shared((block,), "uint8")
            h = T.alloc_shared((block,), "float16")
            f = T.alloc_shared((block,), "float32")
            T.copy(src[cid * block : (cid + 1) * block], s)
            T.tile.cast(h, s, "CAST_NONE", block)
            for i in T.Parallel(block):
                f[i] = T.cast(h[i], "float32") / 127.5 - 1.0
            T.copy(f, dst[cid * block : (cid + 1) * block])

    return main


def main():
    import torch_npu

    torch.npu.set_device(0)
    for m in (67108864, 268431360, 268435456, 273715200):
        src = torch.randint(0, 256, (m,), dtype=torch.uint8, device="npu:0")
        torch.npu.synchronize()
        out = _norm(m)(src)
        torch.npu.synchronize()
        ref = src.cpu().float() / 127.5 - 1.0
        print(
            f"M={m} blocks={m // BLOCK} max_abs={float((out.cpu() - ref).abs().max()):.3e} " f"first={out[:4].tolist()}"
        )


if __name__ == "__main__":
    main()

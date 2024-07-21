import torch
import torch.cuda
import torch.distributed as dist
from flash_attn import flash_attn_varlen_qkvpacked_func
from ring_flash_attn import ring_flash_attn_varlen_qkvpacked_func, zigzag_ring_flash_attn_varlen_qkvpacked_func


def benchmark(f, num_iter=100, forward_only=True, log=True):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    seqlen = 1024 * 8
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    qkv = torch.randn(seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(seqlen, nheads, d, device=device, dtype=dtype)

    cu_seqlens_list = [
        torch.tensor([0, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 256, 7648, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 4096, 8192], device=device, dtype=torch.int32),
        torch.tensor([0, 3104, 6304, 7904, 8064, 8192], device=device, dtype=torch.int32),
    ]
    max_seqlen_list = [(cu_seqlens[1:] - cu_seqlens[:1]).max().item() for cu_seqlens in cu_seqlens_list]

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    if forward_only:
        with torch.no_grad():
            for i in range(num_iter):
                _ = f(
                    qkv,
                    cu_seqlens_list[i % len(cu_seqlens_list)],
                    max_seqlen_list[i % len(max_seqlen_list)],
                    dropout_p=dropout_p,
                    causal=causal,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=deterministic,
                    return_attn_probs=False,
                )
    else:
        for i in range(num_iter):
            qkv.grad = None
            out = f(
                qkv,
                cu_seqlens_list[i % len(cu_seqlens_list)],
                max_seqlen_list[i % len(max_seqlen_list)],
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
                return_attn_probs=False,
            )
            out.backward(dout)
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if rank == 0 and log:
        print(f"{num_iter / time} iter/s, {time} sec")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    forward_only = False

    for f in [
        flash_attn_varlen_qkvpacked_func,
        ring_flash_attn_varlen_qkvpacked_func,
        zigzag_ring_flash_attn_varlen_qkvpacked_func,
    ]:
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__}")
        benchmark(f, forward_only=forward_only, log=False)
        benchmark(f, forward_only=forward_only, log=True)

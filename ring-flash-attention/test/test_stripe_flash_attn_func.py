import random

import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn import stripe_flash_attn_qkvpacked_func


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: " f"max {a.abs().max().item()}, " f"mean {a.abs().mean().item()}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] " f"max {a.abs().max().item()}, " f"mean {a.abs().mean().item()}",
                flush=True,
            )
        dist.barrier()


def extract_local(value, rank, world_size, dim=1):
    value = torch.stack(value.split(world_size, dim=dim), dim=dim).transpose(dim, dim + 1)
    slicer = [rank if i == dim else slice(None) for i in range(len(value.shape))]
    return value[slicer].contiguous()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    set_seed(rank)
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 1
    seqlen = 3824
    nheads = 5
    d = 128
    dropout_p = 0
    causal = True
    deterministic = False

    assert causal
    assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = extract_local(qkv, rank, world_size).detach().clone()
    local_qkv.requires_grad = True
    local_dout = extract_local(dout, rank, world_size).detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = extract_local(out, rank, world_size)
    local_lse = extract_local(lse, rank, world_size, dim=2)

    ring_out, ring_lse, _ = stripe_flash_attn_qkvpacked_func(
        local_qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    out.backward(dout)
    dqkv = qkv.grad

    local_dqkv = extract_local(dqkv, rank, world_size)

    ring_out.backward(local_dout)
    ring_dqkv = local_qkv.grad

    log("local_dq", local_dqkv[:, :, 0, :])
    log("dq diff", local_dqkv[:, :, 0, :] - ring_dqkv[:, :, 0, :])

    log("local_dk", local_dqkv[:, :, 1, :])
    log("dk0 diff", local_dqkv[:, :, 1, :] - ring_dqkv[:, :, 1, :])

    log("local_dv", local_dqkv[:, :, 2, :])
    log("dv diff", local_dqkv[:, :, 2, :] - ring_dqkv[:, :, 2, :])

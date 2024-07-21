import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_attn import flash_attn_qkvpacked_func
from ring_flash_attn import zigzag_ring_flash_attn_qkvpacked_func

from colossalai.shardformer.layer.attn import AttnMaskType, RingAttention


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
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat([value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim)
    return local_value.contiguous()


def run_test(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # or the IP of the master node
    os.environ["MASTER_PORT"] = "8125"  # make sure this port is free
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    set_seed(rank)
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
    extract_local(dout, rank, world_size).detach().clone()

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
    # local_lse = extract_local(lse, rank, world_size, dim=2)
    q, k, v = local_qkv.chunk(3, dim=2)
    q, k, v = [x.squeeze(2).detach().clone().transpose(1, 2) for x in (q, k, v)]
    q.requires_grad = k.requires_grad = v.requires_grad = True
    sp_stream = torch.cuda.Stream()
    sp_group = dist.new_group()
    colo_out = RingAttention.attention(q, k, v, sp_group, sp_stream, AttnMaskType.CAUSAL)

    ring_out, ring_lse, _ = zigzag_ring_flash_attn_qkvpacked_func(
        local_qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    log("colo_out", colo_out, rank0_only=True)
    log("ring_out", ring_out, rank0_only=True)
    # log("lse", lse, rank0_only=True)
    log("colo_out - ring_out", colo_out - ring_out)
    # log("lse diff", local_lse - ring_lse)
    log("ring_out - local_out", ring_out - local_out)
    log("colo_out - local_out", colo_out - local_out)

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# backward:")
        print("#" * 30)

    colo_out.sum().backward()
    qkv.grad
    # q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    colo_dq, colo_dk, colo_dv = [x.transpose(1, 2) for x in (q.grad, k.grad, v.grad)]

    ring_out.sum().backward()
    ring_dqkv = local_qkv.grad
    out.sum().backward()
    dqkv = extract_local(qkv.grad, rank, world_size)

    # log("colo_dq", colo_dq)
    log("dq diff", colo_dq - ring_dqkv[:, :, 0, :])

    # log("colo_dk", colo_dk)
    log("dk diff", colo_dk - ring_dqkv[:, :, 1, :])

    # log("colo_dv", colo_dv)
    log("dv diff", colo_dv - ring_dqkv[:, :, 2, :])
    log("colo_dv - local_dv", colo_dv - dqkv[:, :, 2, :])


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)

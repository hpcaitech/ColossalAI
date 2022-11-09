import math
import os
import socket
import time

import torch
import torch.distributed as dist

from .utils import load_tmp, store_tmp

MB = int((1 << 10) * 1e3)
GB = int((1 << 20) * 1e3)
Byte = 4
FRAMEWORK = 20 / 1e6


def profile(wsize, nbytes, type):
    warmup = 5
    repeat = 25
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    src_device_num = 0
    device = torch.device("cuda", rank)
    hostname = socket.gethostname()

    gpu = f"[{hostname}-{rank}]"

    buf = torch.randn(nbytes // 4).to(device)

    torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        if type == "a":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        elif type == "b":
            dist.broadcast(buf, src=src_device_num)
    torch.cuda.synchronize()

    dist.barrier()
    begin = time.perf_counter()
    for _ in range(repeat):
        if type == "a":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        elif type == "b":
            dist.broadcast(buf, src=src_device_num)
    torch.cuda.synchronize()
    end = time.perf_counter()
    dist.barrier()

    if rank == 0:
        avg_time_s = (end - begin) / repeat - FRAMEWORK
        alg_band = nbytes / avg_time_s
        if type == "b":
            bus_band = alg_band
        elif type == "a":
            bus_band = 2 * (wsize - 1) / wsize * alg_band
        store_tmp(avg_time_s, alg_band)
        print(
            f"{gpu}, Bytes: {nbytes} B,Time: {round(avg_time_s * 1e6,2)} us, Bus bandwidth: {round(bus_band / GB,2)} GB/s"
        )
        return (avg_time_s, alg_band)


def profile_latency(wsize, it=3, type="a"):
    latency = []
    for i in range(it):
        nbytes = int(Byte << i)
        profile(wsize, nbytes, type)
        dist.barrier()
        (t, _) = load_tmp()
        latency.append(t)
    return min(latency)


def profile_bandwidth(wsize, maxbytes, type="a"):
    profile(wsize, maxbytes, type)
    dist.barrier()
    (_, bandwidth) = load_tmp()
    return bandwidth


def profile_ab(rank, wsize, type="a"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29020'
    dist.init_process_group(backend=dist.Backend.NCCL, init_method='env://', world_size=wsize, rank=rank)

    device = torch.device("cuda", rank)
    max_nbytes = torch.tensor(torch.cuda.mem_get_info(device)[0]).to(device)
    max_nbytes = min(int(4 * GB), int(GB << int(math.log2(max_nbytes.item() / GB))))

    if rank == 0:
        print(f"max_nbytes: {max_nbytes} B")

    alpha = profile_latency(wsize, 5)
    beta = 1 / profile_bandwidth(wsize, max_nbytes)
    dist.barrier()

    return (alpha, beta)


def get_one_alpha_beta(rank):
    assert torch.cuda.is_available()
    (alpha, beta) = profile_ab(rank, torch.cuda.device_count(), "a")
    if rank == 0:
        store_tmp(alpha, beta)
        print(f"alpha(us): {round(alpha * 1e6,2)}, beta(us/GB): {round(beta * 1e6 * GB,2)}")

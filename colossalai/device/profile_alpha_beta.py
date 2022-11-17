import fcntl
import math
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MB = int((1 << 10) * 1e3)
GB = int((1 << 20) * 1e3)
Byte = 4
FRAMEWORK = 0
NON_SENSE = (0.1, 0.1)


def printflock(*msgs):
    """ solves multi-process interleaved print problem """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def profile(device1d, nbytes, ctype):
    warmup = 5
    repeat = 25
    rank = dist.get_rank()
    src_device_num = device1d[0]
    wsize = len(device1d)
    group = dist.new_group(device1d)

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    buf = torch.randn(nbytes // 4).to(device)

    torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        if ctype == "a":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
        elif ctype == "b":
            dist.broadcast(buf, src=src_device_num, group=group)
    torch.cuda.synchronize()

    dist.barrier()
    begin = time.perf_counter()
    for _ in range(repeat):
        if ctype == "a":
            dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
        elif ctype == "b":
            dist.broadcast(buf, src=src_device_num, group=group)
    torch.cuda.synchronize()
    end = time.perf_counter()
    dist.barrier()

    if rank == src_device_num:
        avg_time_s = (end - begin) / repeat - FRAMEWORK
        alg_band = nbytes / avg_time_s
        if ctype == "b":
            bus_band = alg_band
        elif ctype == "a":
            bus_band = 2 * (wsize - 1) / wsize * alg_band
        print(
            f"GPU:{rank}, Bytes: {nbytes} B,Time: {round(avg_time_s * 1e6,2)} us, Bus bandwidth: {round(bus_band / GB,2)} GB/s"
        )
        return (avg_time_s, alg_band)
    else:
        return NON_SENSE    # Just a placeholder


def profile_latency(device1d, it=3, ctype="a"):
    latency = []
    for i in range(it):
        nbytes = int(Byte << i)
        (t, _) = profile(device1d, nbytes, ctype)
        latency.append(t)
    return min(latency)


def profile_bandwidth(device1d, maxbytes, ctype="a"):
    (_, bandwidth) = profile(device1d, maxbytes, ctype)
    return bandwidth


def profile_ab(rank, *args):
    wsize = int(torch.cuda.device_count())
    device1d = args[0]
    return_dict = args[1]
    ctype = args[2]
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29020'
    dist.init_process_group(backend=dist.Backend.NCCL, init_method='env://', world_size=wsize, rank=rank)

    device = torch.device("cuda", rank)
    max_nbytes = torch.tensor(torch.cuda.mem_get_info(device)[0]).to(device)
    max_nbytes = min(int(4 * GB), int(GB << int(math.log2(max_nbytes.item() / GB))))

    if rank == device1d[0]:
        print(f"max_nbytes: {max_nbytes} B")

    alpha = profile_latency(device1d, it=5, ctype=ctype)
    beta = 1 / profile_bandwidth(device1d, maxbytes=max_nbytes, ctype=ctype)

    if rank == device1d[0]:
        print(f"alpha(us): {round(alpha * 1e6,2)}, beta(us/GB): {round(beta * 1e6 * GB,2)}")
    return_dict[rank] = (alpha, beta)


def profile_alpha_beta(device1d):
    assert torch.cuda.is_available()
    assert len(device1d) > 0 and len(device1d) <= int(torch.cuda.device_count())

    manager = mp.Manager()
    return_dict = manager.dict()
    ctype = "a"
    mp.spawn(profile_ab, args=[device1d, return_dict, ctype], nprocs=int(torch.cuda.device_count()))
    return return_dict[device1d[0]]

import os
import numpy as np

import torch
from torch.nn import functional as F
from col_fused_softmax_lib import scaled_masked_softmax_forward

def get_latency_for_cuda(func, data, mask, scale):
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300

    for i in range(10):
        func(data, mask, scale)
    
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            func(data, mask, 1)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    return mean_syn


def get_latency_for_torch(func, data):
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300

    for i in range(10):
        func(data, dim=-1)
    
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            func(data, dim=-1)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    return mean_syn
    
def test():
    size = (17, 3, 1024, 256)
    data = torch.randn(size = size, device="cuda", dtype=torch.float16)
    mask = torch.zeros(size = (17, 1, 1024, 256), device="cuda", dtype=torch.uint8)

    out_cuda = scaled_masked_softmax_forward(data, mask, 1)

    out_torch = F.softmax(data, dim = -1)

    torch.allclose(out_cuda.cpu(), out_torch.cpu(), rtol=1e-5, atol=1e-5)

    latency_1 = get_latency_for_cuda(scaled_masked_softmax_forward, data, mask, 1)
    latency_2 = get_latency_for_torch(F.softmax, data)
    print("the cuda implementation is {} ms".format(str(latency_1)))
    print("the original torch cuda implementation is {} ms".format(str(latency_2)))


if __name__ == "__main__":
    test()